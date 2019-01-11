from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import os
import joblib
import numpy as np
import os.path as osp
import copy
import pickle
import sys


def load_from_file(param_pkl_path):
	with open(param_pkl_path, 'rb') as f:
		params= pickle.load(f)
	return params

def save_to_file(param_pkl_path,params):
	with open(param_pkl_path , 'wb') as f:
		pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

def setFromFlat(sess,var_list, flat_params):
	shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
	total_size =np.sum([int(np.prod(shape)) for shape in shapes])
	theta = tf.placeholder(tf.float32, [total_size])
	start = 0
	assigns =[]
	for (shape, v) in zip(shapes, var_list):
		size = int(np.prod(shape))
		assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
		start += size
	op = tf.group(*assigns)
	sess.run(op, {theta: flat_params})

def var_shape(x):
	out = x.get_shape().as_list()
	assert all(isinstance(a, int) for a in out), \
	"shape function assumes that shape is fully known"
	return out

def numel(x):
	return intprod(var_shape(x))
def intprod(x):
	return int(np.prod(x))

def getFlat(sess,var_list):
	op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
	return sess.run(op)






alpha=1.0
def save(sess,save_path):
			ps = sess.run(params)
			joblib.dump(ps, save_path)
def traj_segment_generator(pi, i2, env, horizon, stochastic):
	global alpha
	t = 0
	ac = env.action_space.sample() # not used, just so we have the datatype
	new = True # marks if we're on first timestep of an episode
	ob = env.reset()

	cur_ep_ret = 0 # return in current episode
	cur_ep_len = 0 # len of current episode
	ep_rets = [] # returns of completed episodes in this segment
	ep_lens = [] # lengths of ...

	# Initialize history arrays
	#print("in traj seg ob[0] shape",ob[0].shape)
	obs = np.array([ob[0] for _ in range(horizon)])
	rews = np.zeros(horizon, 'float32')
	vpreds = np.zeros(horizon, 'float32')
	news = np.zeros(horizon, 'int32')
	#print("ac[0] shape",ac[0].shape)
	acs = np.array([ac[0] for _ in range(horizon)])
	prevacs = acs.copy()
	ac_1=ac[0]
	while True:
		prevac = ac_1
		ac_1, vpred = pi.act(stochastic, ob[0])
		# Slight weirdness here because we need value function at time T
		# before returning segment [0, T-1] so we get the correct
		# terminal value
		if t > 0 and t % horizon == 0:
			yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
					"ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
					"ep_rets" : ep_rets, "ep_lens" : ep_lens}
			# Be careful!!! if you change the downstream algorithm to aggregate
			# several of these batches, then be sure to do a deepcopy
			ep_rets = []
			ep_lens = []
			print("alpha",alpha)
		i = t % horizon
		obs[i] = ob[0]
		vpreds[i] = vpred
		news[i] = new
		acs[i] = ac_1
		prevacs[i] = prevac

		#for second agent
		ac_2, _=i2.act(stochastic, ob[1])

		l =[]
		l.append(ac_1)
		l.append(ac_2)
		ac= tuple(i for i in l)

		ob, reww, neww, infos = env.step(ac)
		new=neww[0]
		explore_r=infos[0]['reward_move']
		competition_r=infos[0]['reward_remaining']
		r_1=alpha*explore_r + (1-alpha)*competition_r
		rews[i] = r_1

		cur_ep_ret += r_1
		cur_ep_len += 1
		if new:
			ep_rets.append(cur_ep_ret)
			ep_lens.append(cur_ep_len)
			cur_ep_ret = 0
			cur_ep_len = 0
			ob = env.reset()
			#call add_vtarg_and_adv

		t += 1

def add_vtarg_and_adv(seg, gamma, lam):
	"""
	Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
	"""
	new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
	vpred = np.append(seg["vpred"], seg["nextvpred"])
	T = len(seg["rew"])
	seg["adv"] = gaelam = np.empty(T, 'float32')
	rew = seg["rew"]
	lastgaelam = 0
	for t in reversed(range(T)):
		nonterminal = 1-new[t+1]
		delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
		gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
	seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn, max_timesteps,
		timesteps_per_actorbatch, # timesteps per actor per update
		clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
		optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
		gamma, lam, # advantage estimation
		max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
		callback=None, # you can do anything in the callback, since it takes locals(), globals()
		adam_epsilon=1e-5,
		schedule='constant' # annealing for stepsize parameters (epsilon and adam)
		):
	# Setup losses and stuff
	# ----------------------------------------
	global alpha
	ob_space = env.observation_space.spaces[0]
	ac_space = env.action_space.spaces[0]
	pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
	oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
	atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
	ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

	i2 = policy_fn("i2", ob_space, ac_space)

	lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
	clip_param = clip_param * lrmult # Annealed cliping parameter epislon

	ob = U.get_placeholder_cached(name="ob")
	ac = pi.pdtype.sample_placeholder([None])
	#print("in learn ob ac placeholder shape",ob.shape,ac.shape)
	kloldnew = oldpi.pd.kl(pi.pd)
	ent = pi.pd.entropy()
	meankl = tf.reduce_mean(kloldnew)
	meanent = tf.reduce_mean(ent)
	pol_entpen = (-entcoeff) * meanent

	ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
	surr1 = ratio * atarg # surrogate from conservative policy iteration
	surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
	pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
	vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
	total_loss = pol_surr + pol_entpen + vf_loss
	losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
	loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

	var_list = pi.get_trainable_variables()
	lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
	adam = MpiAdam(var_list, epsilon=adam_epsilon)

	assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
		for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
	compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)
	
	U.initialize()
	adam.sync()

	# Prepare for rollouts
	# ----------------------------------------
	seg_gen = traj_segment_generator(pi,i2, env, timesteps_per_actorbatch, stochastic=True)

	episodes_so_far = 0
	timesteps_so_far = 0
	iters_so_far = 0
	tstart = time.time()
	lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
	rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

	assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

	while True:
		if callback: callback(locals(), globals())
		if max_timesteps and timesteps_so_far >= max_timesteps:
			break
		elif max_episodes and episodes_so_far >= max_episodes:
			break
		elif max_iters and iters_so_far >= max_iters:
			break
		elif max_seconds and time.time() - tstart >= max_seconds:
			break

		if schedule == 'constant':
			cur_lrmult = 1.0
		elif schedule == 'linear':
			cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
		else:
			raise NotImplementedError
		k=int(500)
		alpha=max(1.0 - float(iters_so_far) / k, 0)
		
		logger.log("********** Iteration %i ************"%iters_so_far)

		seg = seg_gen.__next__()
		
		add_vtarg_and_adv(seg, gamma, lam)

		# ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
		ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
		vpredbefore = seg["vpred"] # predicted value function before udpate
		atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
		d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
		optim_batchsize = optim_batchsize or ob.shape[0]

		if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

		assign_old_eq_new() # set old parameter values to new parameter values
		logger.log("Optimizing...")
		logger.log(fmt_row(13, loss_names))
		# Here we do a bunch of optimization epochs over the data
		for _ in range(optim_epochs):
			losses = [] # list of tuples, each of which gives the loss for a minibatch
			for batch in d.iterate_once(optim_batchsize):
				*newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
				adam.update(g, optim_stepsize * cur_lrmult)
				losses.append(newlosses)
			logger.log(fmt_row(13, np.mean(losses, axis=0)))

		logger.log("Evaluating losses...")
		losses = []
		for batch in d.iterate_once(optim_batchsize):
			newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
			losses.append(newlosses)
		meanlosses,_,_ = mpi_moments(losses, axis=0)
		logger.log(fmt_row(13, meanlosses))
		for (lossval, name) in zipsame(meanlosses, loss_names):
			logger.record_tabular("loss_"+name, lossval)
		logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
		lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
		listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
		lens, rews = map(flatten_lists, zip(*listoflrpairs))
		lenbuffer.extend(lens)
		rewbuffer.extend(rews)
		logger.record_tabular("EpLenMean", np.mean(lenbuffer))
		logger.record_tabular("EpRewMean", np.mean(rewbuffer))
		logger.record_tabular("EpThisIter", len(lens))
		episodes_so_far += len(lens)
		timesteps_so_far += sum(lens)
		logger.record_tabular("EpisodesSoFar", episodes_so_far)
		logger.record_tabular("TimestepsSoFar", timesteps_so_far)
		logger.record_tabular("TimeElapsed", time.time() - tstart)
		if MPI.COMM_WORLD.Get_rank()==0:
			logger.dump_tabular()

		save_freq = 1
		if (iters_so_far%save_freq==0):			
			if MPI.COMM_WORLD.Get_rank()==0: 
				path="/home/agrim/baselines/baselines/ppo1/models_7/agent_parameters-v"+str(iters_so_far)+".pkl"
				savepath ="/home/agrim/baselines/baselines/ppo1/save7/agent_parameters-v"+str(iters_so_far)+".pkl"

				U.save_state(savepath)            
				model_1_variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pi")	

				params_path = []
				params = []				
				
				params_path.append(path)
				params.append(getFlat(tf.get_default_session(),model_1_variable))
				save_to_file(params_path[0], params[0])
				
			# load agent 2 policy random 
			if iters_so_far >= save_freq:
				itr=iters_so_far-save_freq
				path = "/home/agrim/baselines/baselines/ppo1/models_7/agent_parameters-v"+str(itr)+".pkl"
				params_path = []
				params_path.append(path)
				params = []	
				params = [load_from_file(param_pkl_path=path) for path in params_path]
				model_2_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="i2")	
				
				setFromFlat(tf.get_default_session(),model_2_variables, params[0])

		iters_so_far += 1	
	return pi

def flatten_lists(listoflists):
	return [el for list_ in listoflists for el in list_]

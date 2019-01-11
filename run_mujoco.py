#!/usr/bin/env python3

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import gym
import gym_compete
import os
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
import numpy as np
from PIL import Image

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




def train(env_id, num_timesteps, seed):
	from baselines.ppo1 import mlp_policy, pposgd_simple
	U.make_session(num_cpu=1).__enter__()
	def policy_fn(name, ob_space, ac_space):
		return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
			hid_size=128, num_hid_layers=2)
	env = make_mujoco_env(env_id, seed)
	pi= pposgd_simple.learn(env, policy_fn,
			max_timesteps=num_timesteps,
			timesteps_per_actorbatch=20480,
			clip_param=0.2, entcoeff=0.0,
			optim_epochs=6, optim_stepsize=1e-3, optim_batchsize=256,
			gamma=0.995, lam=0.95, schedule='linear',
		)
	env.close()
	return pi

def main():
	logger.configure(dir="/home/agrim/baselines/baselines/ppo1/log7")
	parser = mujoco_arg_parser()
	args = parser.parse_args()
	train(args.env,num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
	main()

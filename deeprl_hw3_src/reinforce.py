import deeprl_hw3.imitation;
import deeprl_hw3.reinforce;
import gym
from copy import deepcopy ;
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--alpha',type=float, default=0.00025)
parser.add_argument('--beta', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=4000)
args = parser.parse_args()


with tf.device('/cpu:0'):
	
	env = gym.make('CartPole-v0');
	hard_env = deeprl_hw3.imitation.wrap_cartpole(deepcopy(env));
	policy = deeprl_hw3.reinforce.reinforce(env, alpha = args.alpha, beta = args.beta, max_episodes = args.epochs);
	test_cloned_policy(hard_env, policy, render = R);
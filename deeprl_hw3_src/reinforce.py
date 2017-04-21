import deeprl_hw3.imitation;
import deeprl_hw3.reinforce;
import gym
from copy import deepcopy ;
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--alpha',type=float, default=0.01)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=30000)
args = parser.parse_args()


with tf.device('/cpu:0'):
	
	env = gym.make('CartPole-v0');
	deeprl_hw3.reinforce.reinforce(env,alpha = args.alpha, beta = args.beta, max_episodes = args.epochs);
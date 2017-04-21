import deeprl_hw3.imitation;
import deeprl_hw3.reinforce;
import gym
from copy import deepcopy ;
import tensorflow as tfrl

with tf.device('/cpu:0'):
	
	env = gym.make('CartPole-v0');
	deeprl_hw3.reinforce.reinforce(env);
import argparse
import os
import random
import gym
from gym import wrappers
import numpy as np

#import deeprl_hw2 as tfrl
from deeprl_hw2.preprocessors import *;
env = gym.make('SpaceInvaders-v0')
ob = env.reset();
pre = PreprocessorSequence((84,84),4);
pre.process_state_for_memory(ob);
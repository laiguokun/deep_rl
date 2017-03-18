#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute,Lambda)
#from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

#import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.core import *
from deeprl_hw2.policy import *
from deeprl_hw2.preprocessors import *;

def create_model(window, input_shape, nb_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    #print(nb_actions);
    INPUT_SHAPE = (window,) + input_shape;
    inputs = Input(shape = INPUT_SHAPE);
    #Linear model
    '''
    flatten = Flatten()(inputs);
    outputs = Dense(nb_actions, activation = 'linear')(flatten);
    model = Model(input = inputs, output = outputs);
    '''
    #DQN
    permute = Permute((2,3,1), input_shape = input_shape)(inputs);
    #conv1 = Conv2D(filters = 32, kernel_size = (8,8), strides = 4, activation='relu')(permute);
    #act1 = Activation('relu')(conv1);
    #conv2 = Conv2D(filters = 64, kernel_size = (4,4), strides = 2, activation='relu')(conv1);
    #act2 = Activation('relu')(conv2);
    #conv3 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, activation='relu')(conv2);
    #act3 = Activation('relu')(conv3);
    conv1 = Convolution2D(32, 8,8, subsample = (4, 4), activation='relu')(permute);
    #norm1 = BatchNormalization(axis = 1)(conv1)
    conv2 = Convolution2D(64, 4,4, subsample = (2, 2), activation='relu')(conv1);
    #norm2 = BatchNormalization(axis = 1)(conv2)
    conv3 = Convolution2D(64, 3,3, subsample = (1, 1), activation='relu')(conv2);
    #norm3 = BatchNormalization(axis = 1)(conv3)
    faltten = Flatten()(conv3);
    den1 = Dense(512,activation='relu')(faltten);
    #DQN
    '''
    outputs = Dense(nb_actions, activation = 'linear')(den1);
    '''
    #DQN DUELING
    qtmp = Dense(nb_actions + 1, activation = 'linear')(den1);
    outputs = Lambda(lambda x : K.expand_dims(x[:,0], dim=-1) + x[:,1:] - K.mean(x[:, 1:], keepdims=True),output_shape=(nb_actions,))(qtmp);
    model = Model(input = inputs, output = outputs);
    return model;




def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1


    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=123, type=int, help='Random seed')
    with tf.device('gpu:3'):
        args = parser.parse_args()
        args.input_shape = (84, 84)

        #args.output = get_output_folder(args.output, args.env)

        # here is where you should start up a session,
        # create your DQN agent, create your model, etc.
        # then you can run your fit method.
        env = gym.make(args.env)
        env = wrappers.Monitor(env, 'tmp/SpaceInvader-experiment-1',force=True)
        np.random.seed(args.seed)
        env.seed(args.seed);
        nb_actions = env.action_space.n
        q_network = create_model(4, args.input_shape, nb_actions, model_name='q_network')
        memory = ReplayMemory(max_size = 1000000);
        policy = GreedyEpsilonPolicy();
        preprocessor = PreprocessorSequence((84,84),4);

        dqn = DQNAgent(q_network, preprocessor, memory, policy, nb_actions, num_burn_in=5000, enable_double_dqn = True, enable_double_dqn_hw=False, reward_record=open('double_dqn_reward.txt','w'), loss_record=open('double_dqn_loss.txt','w')) ;
        dqn.compile(Adam(lr=.0001), mean_huber_loss)
        dqn.fit(env, 5000000)
        dqn.evaluate(env, 100)
        env.close()
        gym.upload('tmp/SpaceInvader-experiment-1', api_key='sk_0Z6MMPCTgiAGwmwJ54zLQ')
        q_network.save('dqn_cnn1.h5')

if __name__ == '__main__':
    main()

import gym
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute,Lambda)
from keras.models import Model, Sequential
import deeprl_hw3.imitation;
def get_total_reward(env, model):
    """compute total reward

    Parameters
    ----------
    env: gym.core.Env
      The environment. 
    model: (your action model, which can be anything)

    Returns
    -------
    total_reward: float
    """
    pass

def sample_episode(env, policy):
    rewards = []
    states = [];
    actions = [];
    state = env.reset()
    is_done = False
    while not is_done:
        states.append(state);
        action = np.argmax(policy.predict_on_batch(state[np.newaxis, ...])[0])
        actions.append(action);
        state, reward, is_done, _ = env.step(action)
        rewards.append(reward)
    return states, actions, rewards 


def build_model():
    """ build the cloned model
    """
    INPUT_SHAPE = (4,);
    inputs = Input(shape = INPUT_SHAPE);
    hidden1 = Dense(16, activation = 'relu')(inputs);
    hidden2 = Dense(16, activation = 'relu')(hidden1);
    hidden3 = Dense(16, activation = 'relu')(hidden2);
    Pi = Dense(2, activation = 'softmax')(hidden3)
    V = Dense(1, activation = 'linear')(hidden3)
    policy = Model(inputs, Pi);
    value = Model(inputs, V);
    return policy, value; 


def choose_action(model, observation):
    """choose the action 

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation

    Returns
    -------
    p: float 
        probability of action 1
    action: int
        the action you choose
    """
    p = expert.predict_on_batch(np.asarray([observation]))[0];
    action = np.argmax(p);
    return p[1], action


def reinforce(env, alpha = 0.01, beta = 0.01, max_episodes = 1000):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    Keras Model 
    """

    policy, value = build_model();
    policy_params = policy.trainable_weights;
    value_params = value.trainable_weights;
    value_output = value.output;
    action_output = tf.log(policy.output);
    value_gradient = tf.gradients(value_output, value_params)
    policy_gradient = [0, 0]
    policy_gradient[0] = tf.gradients(action_output[0], policy_params)
    policy_gradient[1] = tf.gradients(action_output[1], policy_params)
    count = 0;
    while (True):
        states, actions, rewards = sample_episode(env, policy);
        for i in range(len(states)):
            Gt = rewards[i];
            delta = Gt - value.predict_on_batch(np.asarray([states[i]]))[0];
            for x in range(len(value_params)):
                value_params[x] += beta * delta * value_gradient[x];
            for x in range(len(policy_params)):
                policy_params[x] += alpha * delta * policy_gradient[actions[i]][x];
            #policy_params += alpha * delta * policy_gradient[action[i]];
        if (count % 20 == 0):
            deeprl_hw3.imitation.test_cloned_policy(env, policy, num_episodes=50, render=False)
        count += 1;
        if (count > max_episodes):
            break;
    return 0

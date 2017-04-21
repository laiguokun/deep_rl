import gym
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute,Lambda)
from keras.models import Model, Sequential
import deeprl_hw3.imitation;
from keras import backend as K
from copy import deepcopy 
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
        states.append(deepcopy(state));
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

import time;
def reinforce(env, alpha = 0.1, beta = 0.1, max_episodes = 30000):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    Keras Model 
    """
    m1 = np.matrix('1;0');
    m2 = np.matrix('0;1');
    policy, value = build_model();
    policy_params = policy.trainable_weights;
    value_params = value.trainable_weights;
    value_output = value.output;
    log_output = tf.log(policy.output);
    policy_output = [log_output * m1, log_output * m2]
    value_gradient = tf.gradients(value_output, value_params)
    policy_gradient = [0, 0]
    policy_gradient[0] = tf.gradients(policy_output[0], policy_params)
    policy_gradient[1] = tf.gradients(policy_output[1], policy_params)
    count = 0;
    get_vgrad = K.function([value.layers[0].input], value_gradient);
    get_pgrad = [0,0];
    get_pgrad[0] = K.function([policy.layers[0].input], policy_gradient[0]);
    get_pgrad[1] = K.function([policy.layers[0].input], policy_gradient[1]);
    while (True):
        
        states, actions, rewards = sample_episode(env, policy);
        #print(len(states));
        for i in range(len(states)):
            Gt = rewards[i];
            delta = Gt - value.predict_on_batch(np.asarray([states[i]]))[0];
            v_grads = get_vgrad([np.asarray([states[i]])]);
            p_grads = get_pgrad[actions[i]]
            p_grads = p_grads([np.asarray([states[i]])])
            for x in range(len(value_params)):
                value_params[x] += beta * delta * v_grads[x];
            for x in range(len(policy_params)):
                policy_params[x] += alpha * delta * p_grads[x];
        if (count % 60 == 0):
            deeprl_hw3.imitation.test_cloned_policy(env, policy, num_episodes=50, render=False)
            policy.save('policy.h5');
            value.save('value.h5');
        count += 1;
        if (count > max_episodes):
            break;
    return 0

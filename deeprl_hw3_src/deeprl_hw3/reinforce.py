import gym
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute,Lambda)
from keras.models import Model, Sequential
import deeprl_hw3.imitation;
from keras import backend as K
from copy import deepcopy 
def get_total_reward(env, model, num_episodes = 100):
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
    total_rewards = []

    for i in range(num_episodes):
        #print('Starting episode {}'.format(i))
        total_reward = 0
        state = env.reset()
        is_done = False
        while not is_done:
            action = choose_action(model, state);
            state, reward, is_done, _ = env.step(action)
            total_reward += reward
        #print('Total reward: {}'.format(total_reward))
        total_rewards.append(total_reward)

    print('Average total reward: {} (std: {}). min: {}, max: {}'.format(
        np.mean(total_rewards), np.std(total_rewards), np.min(total_rewards), np.max(total_rewards)));

def sample_episode(env, policy):
    rewards = []
    states = [];
    actions = [];
    state = env.reset()
    is_done = False
    while not is_done:
        states.append(deepcopy(state));
        action = choose_action(policy, state);
        actions.append(action);
        state, reward, is_done, _ = env.step(action)
        rewards.append(reward)
    for i in range(len(rewards)-1):
        x = len(rewards) - i - 2;
        rewards[x] = rewards[x] + rewards[x+1];
    return states, actions, rewards 


def build_model(alpha, beta):
    """ build the cloned model
    """
    eplison = 1e-3
    INPUT_SHAPE = (4,);
    inputs = Input(shape = INPUT_SHAPE);
    hidden1 = Dense(16, activation = 'relu')(inputs);
    hidden2 = Dense(16, activation = 'relu')(hidden1);
    hidden3 = Dense(16, activation = 'relu')(hidden2);
    Pi = Dense(2, activation = 'softmax')(hidden3)

    v_hidden1 = Dense(16, activation = 'relu')(inputs);
    v_hidden2 = Dense(16, activation = 'relu')(v_hidden1)
    V = Dense(1, activation = 'linear')(v_hidden1)
    policy = Model(inputs, Pi);
    value = Model(inputs, V);

    delta = tf.placeholder(dtype=tf.float32, name="delta")
    action = tf.placeholder(dtype=tf.int32, name="action")

    policy_params = policy.trainable_weights;
    value_params = value.trainable_weights;
    value_output = value.output[0];
    policy_output = tf.gather(policy.output[0], action)
    policy_output = tf.log(policy_output+eplison);

    value_gradient = tf.gradients(value_output, value_params)
    policy_gradient = tf.gradients(policy_output, policy_params)

    v_updates = [ K.update(param, param + beta * delta * gparam) for param, gparam in zip(value_params, value_gradient)];
    p_updates = [ K.update(param, param + alpha * delta * gparam) for param, gparam in zip(policy_params, policy_gradient)];
    value_update = K.function([value.layers[0].input, delta], value_gradient, updates = v_updates);
    #tmp = policy_params[0] + alpha * delta * policy_gradient[0];
    policy_update = K.function([policy.layers[0].input, delta, action], policy_gradient, updates = p_updates);

    return policy, value, policy_update, value_update; 



def choose_action(model, observation, eplison = 0.05):
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
    if (np.random.random() < eplison):
        return np.random.randint(2);
    p = model.predict_on_batch(np.asarray([observation]))[0];
    action = np.random.choice(np.arange(len(p)), p=p)
    return action

import time;
def reinforce(env, alpha = 0.01, beta = 0.01, max_episodes = 30000):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    Keras Model 
    """
    policy, value, policy_update, value_update = build_model(alpha, beta);
    count = 0;
    #sess = tf.InteractiveSession()
    while (True):
        #x = time.time()
        states, actions, rewards = sample_episode(env, policy);
        #print(time.time() - x)
        for i in range(len(states)):
            Gt = rewards[i];
            inputs = np.asarray([states[i]]);
            delta = Gt - (200.0-i) * value.predict_on_batch(inputs)[0];
            #print(delta);
            value_update([inputs, delta]);
            policy_update([inputs, delta, actions[i]])
        count += 1;
        if (count % 100 == 0):
            #print(alpha * delta * tmp)
            #print(value.trainable_weights[0].eval());
            #print(policy.trainable_weights[0].eval());
            get_total_reward(env, policy, num_episodes=50)
            policy.save('policy.h5');
            value.save('value.h5');
        if (count % 1000 == 0):
            #alpha *= 0.5;
            #beta *= 0.5;
            print(alpha, beta);

        if (count > max_episodes):
            break;
    return 0

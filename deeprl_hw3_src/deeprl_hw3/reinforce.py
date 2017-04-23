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
    return np.mean(total_rewards);

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
        if (len(states)>=200):
            break;
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

    policy = Model(inputs, Pi);

    v_hidden1 = Dense(16, activation = 'relu')(inputs);
    v_hidden2 = Dense(16, activation = 'relu')(v_hidden1);
    V = Dense(1, activation = 'linear')(v_hidden2)

    value = Model(inputs, V);

    value.compile(optimizer='adam', loss='mean_squared_error');

    #delta = tf.placeholder(dtype=tf.float32, name="delta")
    #action = tf.placeholder(dtype=tf.int32, name="action")
    delta = Input(name='delta', shape=(1,))
    action = Input(name='action', shape=(1,), dtype='int32')

    def output_layer(args):
        policy_output, delta, action = args
        policy_output = tf.gather(policy.output[0], action)
        policy_output = - delta * tf.log(policy_output)
        return policy_output;

    #y_true = Input(name='y_true', shape=(1,));

    loss = Lambda(output_layer, output_shape=(1,), name='loss')([policy.output, delta, action]);

    trainable_policy = Model(input=[inputs, delta, action], output=[loss])

    loss_for_model = [
        lambda y_true, y_pred: y_pred, 
    ]
    trainable_policy.compile(optimizer='adam', loss=loss_for_model);


    #policy_params = policy.trainable_weights;
    #value_gradient = tf.gradients(value_output, value_params)
    #policy_gradient = tf.gradients(policy_output, policy_params)

    #v_updates = [ K.update(param, param + beta * delta * gparam) for param, gparam in zip(value_params, value_gradient)];
    #p_updates = [ K.update(param, param - alpha * gparam) for param, gparam in zip(policy_params, policy_gradient)];
    #value_update = K.function([value.layers[0].input, delta], value_gradient, updates = v_updates);
    #policy_update = K.function([policy.layers[0].input, delta, action], policy_gradient, updates = p_updates);
    #optimizer = tf.train.AdamOptimizer(0.01).minimize(policy_output)

    return policy, trainable_policy, value#, policy_update; 



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
    p = model.predict_on_batch(np.asarray([observation]))[0];
    action = np.random.choice(np.arange(len(p)), p=p)
    return action

import time;
def reinforce(env, alpha = 0.01, beta = 0.99, max_episodes = 3000):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    Keras Model 
    """
    #policy, policy_update = build_model(alpha, beta);
    policy, trainable_policy, value = build_model(alpha, beta)
    count = 0;
    cnt = 0;
    print(alpha, beta);
    while (True):
        states, actions, rewards = sample_episode(env, policy);
        for i in range(len(states)):
            Gt = rewards[i];
            inputs = np.asarray([states[i]]);
            reward = Gt * (beta ** i);
            predict_reward = value.predict_on_batch([inputs])[0]
            delta = np.asarray([reward - (200.0-i)* predict_reward]);
            value.train_on_batch([inputs],[np.asarray(delta/(200.0-i))])
            action = np.asarray([actions[i]]);
            trainable_policy.train_on_batch([inputs, delta, action],[action])
            #trainable_policy.train_on_batch([inputs, delta, actions[i]],[actions[i]])
        count += 1;
        if (count % 100 == 0):
            if (get_total_reward(env, policy, num_episodes=50) == 200.0):
                cnt += 1;
            policy.save('policy.h5');
            if (cnt >= 3):
                break;
        '''
        if (count % 1000 == 0):
            alpha *= 0.1;
            print(alpha, beta);
        '''
        if (count > max_episodes):
            break;
    return policy

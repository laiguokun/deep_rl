"""Functions for imitation learning."""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from keras.models import model_from_yaml
import numpy as np
import time
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute,Lambda)
from keras.models import Model, Sequential
from copy import deepcopy 

def load_model(model_config_path, model_weights_path=None):
    """Load a saved model.

    Parameters
    ----------
    model_config_path: str
      The path to the model configuration yaml file. We have provided
      you this file for problems 2 and 3.
    model_weights_path: str, optional
      If specified, will load keras weights from hdf5 file.

    Returns
    -------
    keras.models.Model
    """
    with open(model_config_path, 'r') as f:
        model = model_from_yaml(f.read())

    if model_weights_path is not None:
        model.load_weights(model_weights_path)

    model.summary()

    return model


def generate_expert_training_data(expert, env, num_episodes=100, render=True):
    """Generate training dataset.

    Parameters
    ----------
    expert: keras.models.Model
      Model with expert weights.
    env: gym.core.Env
      The gym environment associated with this expert.
    num_episodes: int, optional
      How many expert episodes should be run.
    render: bool, optional
      If present, render the environment, and put a slight pause after
      each action.

    Returns
    -------
    expert_dataset: ndarray(states), ndarray(actions)
      Returns two lists. The first contains all of the states. The
      second contains a one-hot encoding of all of the actions chosen
      by the expert for those states.
    """

    states = [];
    actions = [];
    for i in range(num_episodes):
        state = env.reset();
        is_terminal = False;
        while not is_terminal:
            action = expert.predict_on_batch(np.asarray([state]))[0];
            #print(action);
            action = np.argmax(action);
            states.append(deepcopy(state))
            action_one_hot = np.zeros(2);
            action_one_hot[action] = 1;
            actions.append(action_one_hot);
            state, reward, is_terminal, info = env.step(action);
            #break;
    return np.asarray(states), np.asarray(actions);


def build_cloned_model():
    """ build the cloned model
    """
    INPUT_SHAPE = (4,);
    model = Sequential();
    model.add(Dense(16, activation = 'relu', input_shape = INPUT_SHAPE));
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))

    return model; 

def dagger(model, expert, env, train_states, train_actions, num_episodes = 20):
    """ train a clonded model based on the dagger algorithm
    """
    for i in range(num_episodes):
        model.fit(train_states, train_actions, verbose =False, epochs = 50);
        #new data
        new_states, new_actions = generate_expert_training_data(model, env, num_episodes = 1);
        new_actions = [];
        for state in new_states:
            action = expert.predict_on_batch(np.asarray([state]))[0];
            action = np.argmax(action);
            action_one_hot = np.zeros(2);
            action_one_hot[action] = 1;
            new_actions.append(action_one_hot);
        new_actions = np.asarray(new_actions);
        train_states = np.concatenate((train_states, new_states), axis = 0)
        train_actions = np.concatenate((train_actions, new_actions), axis = 0);
        if (i % 1 == 0):
            test_cloned_policy(env, model, num_episodes=100, render=False)

def test_cloned_policy(env, cloned_policy, num_episodes=100, render=True):
    """Run cloned policy and collect statistics on performance.

    Will print the rewards for each episode and the mean/std of all
    the episode rewards.

    Parameters
    ----------
    env: gym.core.Env
      The CartPole-v0 instance.
    cloned_policy: keras.models.Model
      The model to run on the environment.
    num_episodes: int, optional
      Number of test episodes to average over.
    render: bool, optional
      If true, render the test episodes. This will add a small delay
      after each action.
    """
    total_rewards = []

    for i in range(num_episodes):
        #print('Starting episode {}'.format(i))
        total_reward = 0
        state = env.reset()
        if render:
            env.render()
            time.sleep(.1)
        is_done = False
        while not is_done:
            action = np.argmax(
                cloned_policy.predict_on_batch(state[np.newaxis, ...])[0])
            state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
                time.sleep(.1)
        #print('Total reward: {}'.format(total_reward))
        total_rewards.append(total_reward)

    print('Average total reward: {} (std: {}). min: {}, max: {}'.format(
        np.mean(total_rewards), np.std(total_rewards), np.min(total_rewards), np.max(total_rewards)));


def wrap_cartpole(env):
    """Start CartPole-v0 in a hard to recover state.

    The basic CartPole-v0 starts in easy to recover states. This means
    that the cloned model actually can execute perfectly. To see that
    the expert policy is actually better than the cloned policy, this
    function returns a modified CartPole-v0 environment. The
    environment will start closer to a failure state.

    You should see that the expert policy performs better on average
    (and with less variance) than the cloned model.

    Parameters
    ----------
    env: gym.core.Env
      The environment to modify.

    Returns
    -------
    gym.core.Env
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_reset = unwrapped_env._reset

    def harder_reset():
        unwrapped_env.orig_reset()
        unwrapped_env.state[0] = np.random.choice([-1.5, 1.5])
        unwrapped_env.state[1] = np.random.choice([-2., 2.])
        unwrapped_env.state[2] = np.random.choice([-.17, .17])
        return unwrapped_env.state.copy()

    unwrapped_env._reset = harder_reset

    return env

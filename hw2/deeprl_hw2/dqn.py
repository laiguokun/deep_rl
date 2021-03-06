"""Main DQN agent."""
from utils import *;
from keras.layers import Lambda, Input, merge, Layer, Dense
from keras.models import Model
from copy import deepcopy
import tensorflow as tf
import numpy as np;
import keras.backend as K

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 nb_actions,
                 gamma = 0.99,
                 target_update_freq = 10000,
                 num_burn_in = 50000,
                 train_freq = 1,
                 batch_size = 32,
                 enable_double_dqn = False,
                 enable_double_dqn_hw = False,
                 reward_record = None,
                 loss_record = None):

        self.q_network = q_network;
        self.preprocessor = preprocessor;
        self.memory = memory;
        self.policy = policy;
        self.nb_actions = nb_actions;
        self.gamma = gamma;
        self.target_update_freq = target_update_freq;
        self.num_burn_in = num_burn_in;
        self.train_freq = train_freq;
        self.batch_size = batch_size;
        self.count = 0;
        self.output = 0;
        self.episode = 0;
        self.enable_double_dqn = enable_double_dqn;
        self.enable_double_dqn_hw = enable_double_dqn_hw;
        self.eval_preprocessor = preprocessor;
        self.trainable_target_model = None;
        self.reward_record = reward_record;
        self.loss_record = loss_record;

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        #with target fixed
        self.target_model = clone_model(self.q_network);
        self.target_model.compile(optimizer='sgd', loss=loss_func);
        #without target fixed
        #self.target_model = self.q_network;
        
        self.q_network.compile(optimizer='sgd', loss=loss_func);
        y_pred = self.q_network.output;
        y_true = Input(name='y_true', shape=(self.nb_actions,));
        mask = Input(name='mask', shape=(self.nb_actions,))

        def loss_f(args):
            y_true, y_pred, mask = args;
            return loss_func(y_true, y_pred*mask);

        loss = Lambda(loss_f, output_shape=(1,), name='loss')([y_true, y_pred, mask]);
        self.trainable_model = Model(input=[self.q_network.input, y_true, mask], output=[loss])

        loss_for_model = [
            lambda y_true, y_pred: y_pred,  # loss is in output
            #lambda y_true, y_pred: K.zeros_like(y_pred),  # only use for debugging
        ]

        self.trainable_model.compile(optimizer=optimizer, loss=loss_for_model);
        
        if (self.enable_double_dqn_hw):
            y_pred = self.target_model.output;

            loss = Lambda(loss_f, output_shape=(1,), name='loss')([y_true, y_pred, mask]);
            self.trainable_target_model = Model(input=[self.target_model.input, y_true, mask], output=[loss])

            self.trainable_target_model.compile(optimizer=optimizer, loss=loss_for_model);


    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        q_values = self.q_network.predict_on_batch(np.asarray([state]));
        return q_values.flatten();

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        q_values = self.calc_q_values(state);
        action = self.policy.select_action(self.train, self.count, q_values=q_values);
        return action;

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        pass

    def training(self):
        if self.count < self.num_burn_in:
            return;
        
        if self.enable_double_dqn_hw:
            r = np.random.random_sample;
            if r < 0.5:
                self.target_model, self.q_network = self.q_network, self.target_model;
                self.trainable_target_model, self.trainable_model = self.trainable_model, self.trainable_target_model;
                
        experiences = self.memory.sample(self.batch_size);
        s1_batch = [];
        reward_batch = [];
        action_batch = [];
        terminal_batch = []
        s2_batch = []
        for item in experiences:
            s1_batch.append(item.s1);
            s2_batch.append(item.s2);
            reward_batch.append(item.r);
            action_batch.append(item.a);
            terminal_batch.append(0. if item.is_terminal else 1.);

        s1_batch = np.asarray(s1_batch);
        s1_batch = self.preprocessor.process_batch(s1_batch);
        s2_batch = np.asarray(s2_batch);
        s2_batch = self.preprocessor.process_batch(s2_batch);
        reward_batch = np.asarray(reward_batch);
        action_batch = np.asarray(action_batch);
        terminal_batch = np.asarray(terminal_batch);
        if (self.enable_double_dqn):
            q_value = self.q_network.predict_on_batch(s2_batch);
            actions = np.argmax(q_value, axis = 1)
            qstar_values = self.target_model.predict_on_batch(s2_batch);
            qstar_batch = qstar_values[range(self.batch_size), actions];
        else:
            qstar_values = self.target_model.predict_on_batch(s2_batch);
            qstar_batch = np.max(qstar_values, axis=1).flatten();
            
        R = reward_batch + self.gamma * qstar_batch * terminal_batch;
        y_true = np.zeros((self.batch_size, self.nb_actions));
        mask = np.zeros((self.batch_size, self.nb_actions));
        for i in range(self.batch_size):
            y_true[i][action_batch[i]] = R[i];
            mask[i][action_batch[i]] = 1.;

        res = self.trainable_model.train_on_batch([s1_batch, y_true, mask], [R])
        self.output = self.output+res;
        if (self.count % 1000 == 0):
            print(self.count, self.output/1000);
            if (self.loss_record != None):
                self.loss_record.write(str(self.output/1000) + '\n');
                self.loss_record.flush();
            self.output = 0;
        #with target fixed
        if (self.count % self.target_update_freq == 0 and self.enable_double_dqn_hw == False):
            get_hard_target_model_updates(self.target_model, self.q_network);
        


    def fit(self, env, num_iterations, action_repete=1, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
         #number of iteration
        self.train = True;
        observation = None;
        R = None;
        step = None;
        #print('eval reward', self.evaluate(self.env, 5, 10000));        
        is_terminal = False
        cc = 0;
        while (cc < num_iterations or not is_terminal):
            if (observation is None): # start a new episode
                step = 0;
                R = 0.;
                observation = deepcopy(env.reset());
                self.preprocessor.reset();
                self.state = self.preprocessor.process_state_for_memory(observation);
            r = 0.;
            action = self.select_action(self.state);
            self.prev_state = self.state;
            for _ in range(action_repete):
                observation, reward, is_terminal, info = env.step(action);
                R += reward;
                reward = self.preprocessor.process_reward(reward);
                r += reward;
                observation = deepcopy(observation);
                self.state = self.preprocessor.process_state_for_memory(observation);
                if is_terminal:
                    break;
                
            #if self.count < 1000:
            self.memory.append(self.prev_state, action, r, self.state, is_terminal);
            self.training();
            self.count += 1;
            cc += 1;
            if (is_terminal):
                observation = None;
                self.episode +=1;
                print(self.episode)
                if (self.reward_record != None):
                    self.reward_record.write('training '+ str(R) + ' ' + str(self.count) + '\n');
                    self.reward_record.flush();
                print(R);




    def evaluate(self, env, num_episodes, action_repete = 1, policy = None, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        self.train = False;
        R = 0;
        if (policy == None):
            policy = self.policy
        for episode in range(num_episodes):
            r = 0;
            step = 0;
            state = None;
            observation = deepcopy(env.reset());
            self.eval_preprocessor.reset();
            state = self.eval_preprocessor.process_state_for_memory(observation);
            is_terminal = False;
            while not is_terminal:
                action = self.select_action(state);
                for _ in range(action_repete):
                    observation, reward, is_terminal, info = env.step(action);
                    observation = deepcopy(observation);
                    state = self.eval_preprocessor.process_state_for_memory(observation);
                    r += reward;
                    step += 1;
                    if is_terminal:
                        break;
            R += r;
            if (self.reward_record != None):
                self.reward_record.write('evaluate ' + str(r) + ' ' + str(self.count) + '\n');
                self.reward_record.flush();
        print('eval result', R/num_episodes);
        return R/num_episodes;


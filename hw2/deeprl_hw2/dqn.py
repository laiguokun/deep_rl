"""Main DQN agent."""
from utils import *;
from keras.layers import Lambda, Input, merge, Layer, Dense
from keras.models import Model
from copy import deepcopy

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
                 num_burn_in = 1000,
                 train_freq = 1,
                 batch_size = 32):

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
        self.target_model = clone_model(self.q_network);
        self.target_model.compile(optimizer='sgd', loss=loss_func);
        y_pred = self.q_network.output;
        y_true = Input(name='y_true', shape=(self.nb_actions,));
	    mask = Input(name='mask', shape=(self.nb_actions,))

	    def loss_f(args):
	       y_true, y_pred, mask = args;
	       return loss_func(y_true, y_pred) * mask;

        loss = Lambda(loss_f, output_shape=(1,), name='loss')([y_true, y_pred, mask]);
        self.trainable_model = Model(input=[self.q_network.input, y_true, mask], output=[loss])

        loss_for_model = [
            lambda y_true, y_pred: y_pred,  # loss is in output
            #lambda y_true, y_pred: K.zeros_like(y_pred),  # only use for debugging
        ]

        self.trainable_model.compile(optimizer=optimizer, loss=loss_for_model);


    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        q_values = self.model.predict_on_batch(state);
        return q_values;

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
        action = self.policy.select_action(q_values=q_values);
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

    def append_memory(self, observation, action, reward):
        # add an observation to the replay memory
        self.state = self.preprocessor.new_observation(observation);
        self.memory.append(self.prev_state, action, reward, self.state, is_terminal);

    def training(self):
        if self.count < self.num_burn_in:
            return;
        experiences = self.memory.sample(self.batch_size);
        s1_batch = [];
        reward_batch = [];
        action_batch = [];
        s2_batch = []
        for item in experiences:
            s1_batch.append(item.s1);
            s2_batch.append(item.s2);
            reward_batch.append(item.r);
            action_batch.append(item.a);

        s1_batch = np.asarray(s1_batch);
        s2_batch = np.asarray(s2_batch);
        reward_batch = np.asarray(reward_batch);
        action_batch = np.asarray(action_batch);
        qstar_values = self.target_model.predict_on_batch(s1_batch);
        qstar_batch = np.max(qstar_values, axis=1);
        R = reward + self.gamma * qstar_batch
        y_true = np.zeros((self.batch_size, self.nb_actions));
        mask = np.zeros((self.batch_size, self.nb_actions));
        for i in range(self.batch_size):
            y_true[i][action_batch[i]] = R[i];
            mask[i][action_batch[i]] = R[i];

        res = self.trainable_model.train_on_batch([s1_batch, y_true, mask], [R])
        if (self.count & self.target_update_freq == 0):
            get_hard_target_model_updates(self.target_model, self.model);
        


    def fit(self, env, num_iterations, max_episode_length=None):
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

        self.count = 0 #number of iteration

        observation = None;
        reward = None;
        step = None;
        self.preprocessor.clear();
        episode = 0;
        while (self.count < num_iterations):
            if (observation is None): # start a new episode
                step = 0;
                reward = 0.;
                observation = deepcopy(env.reset());
                self.state = self.preprocessor.reset(observation);

            action = self.action_from_state(observation);
            self.prev_state = self.state;
            observation, r, is_terminal, info = env.step(action);
            observation = deepcopy(observation);
            reward += r;
            self.append_memory(observation, action, r, is_terminal);
            self.training();
            self.count += 1;

            if (is_terminal):
                observation = None;
                episode +=1;




    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        pass

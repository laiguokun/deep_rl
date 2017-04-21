import deeprl_hw3.imitation;
import gym
from copy import deepcopy ;

expert = deeprl_hw3.imitation.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f');
expert.compile(optimizer='sgd', loss='mse')
env = gym.make('CartPole-v0');
hard_env = deeprl_hw3.imitation.wrap_cartpole(deepcopy(env));
train_data = [];
train_steps = [1,10,50,100];
train_steps = [1];
for i in range(len(train_steps)):
	train_data.append(deeprl_hw3.imitation.generate_expert_training_data(expert, env, train_steps[i]));

R = False;
print('experted model in original setting:');
deeprl_hw3.imitation.test_cloned_policy(env, expert, render = R);
print('experted model in harder setting:');
deeprl_hw3.imitation.test_cloned_policy(hard_env, expert, render = R);
'''
for i in range(len(train_steps)):
	print('eposide:', train_steps[i])
	model = deeprl_hw3.imitation.build_cloned_model();
	model.compile(optimizer='adam', loss='binary_crossentropy');
	model.fit(train_data[i][0],train_data[i][1], verbose =False, epochs = 100)
	print('cloned model in original setting:')
	deeprl_hw3.imitation.test_cloned_policy(env, model, render = R);
	print('cloned model in harder setting:')
	deeprl_hw3.imitation.test_cloned_policy(hard_env, model, render = R);
'''

#DAGGER
model = deeprl_hw3.imitation.build_cloned_model();
model.compile(optimizer='adam', loss='binary_crossentropy');
deeprl_hw3.imitation.dagger(model, expert, env, train_data[0][0], train_data[0][1]);
print('dagger model in harder settting:')
deeprl_hw3.imitation.test_cloned_policy(hard_env, model, render = R)
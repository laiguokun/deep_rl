import deeprl_hw3.imitation;
import gym

expert = deeprl_hw3.imitation.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f');
expert.compile(optimizer='sgd', loss='mse')
env = gym.make('CartPole-v0');
hard_env = deeprl_hw3.imitation.warp_cartpole(env);
train_data = [];
train_steps = [1,10,50,100];
for i in range(train_steps):
	train_data.append(deeprl_hw3.imitation.generate_expert_training_data(expert, env, train_steps[i]));

print('experted model in original setting:', deeprl_hw3.imitation.test_cloned_policy(env, expert));
print('experted model in harder setting:', deeprl_hw3.imitation.test_cloned_policy(hard_env, expert));
for i in range(train_steps):
	model = deeprl_hw3.imitation.build_cloned_model();
	model.compile(optimizer='adam', loss='binary_crossentropy', epochs = 100);
	model.fit(train_data[i][0],train_data[i][1])
	print('cloned model in original setting:', deeprl_hw3.imitation.test_cloned_policy(env, model));
	print('cloned model in harder setting:', deeprl_hw3.imitation.test_cloned_policy(hard_env, model));
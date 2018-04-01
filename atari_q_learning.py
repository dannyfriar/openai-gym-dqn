# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import sys
import os
import math
import pdb
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import OrderedDict
from datetime import datetime
from skimage.color import rgb2gray
from skimage.transform import resize

##------------------------------------------------------------
def progress_bar(value, endvalue, bar_length=20):
    """Print progress bar to the console"""
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def save_results(results_list, col_names, file_name):
	"""Pass list of results, column_names to save to file_name (must be CSV)"""
	output_data = OrderedDict()
	for idx, name in enumerate(col_names):
		output_data[name] = results_list[idx]
	df = pd.DataFrame.from_dict(output_data)

	# Move file to temp folder if it exists already
	if os.path.exists(file_name):
		new_name = file_name.replace('results', 'results/temp')
		new_name = new_name.replace('.csv', '%s.csv' % str(datetime.now().microsecond))
		os.rename(file_name, new_name)

	df.to_csv(file_name, index=False)


##------------------------------------------------------------
##----------------- Game functions ---------------------------
def get_init_state(env):
	"""Initialize the state for an environment i.e. first 4 frames"""
	state_list = []
	observation = env.reset()
	for _ in range(4):
		state_list.append(preprocess(observation))
		action = env.action_space.sample()
		observation, _, _, info = env.step(action)
	state = np.stack(state_list, axis=2)
	return env, state, observation

def update_state(state, observation):
	"""Remove first obs and add latest obs (latest 4 frames for state)"""
	state = np.append(state, preprocess(observation).reshape(28, 28, 1), axis=2)
	new_state = state[:, :, 1:]
	return new_state

def clip_rewards(reward):
	"""Clip rewards to be {-1, 0, 1}"""
	if reward > 0:
		return 1
	elif reward < 0:
		return -1
	else:
		return 0

def preprocess(observation):
	""" Convert the 4 210x160x3 uint8 frames into a single agent state, size 28x28x4"""
	resized = resize(observation, (28, 28), preserve_range=True)
	return rgb2gray(resized).astype("uint8")
	# return cv2.cvtColor(cv2.resize(observation, (28, 28)), cv2.COLOR_BGR2GRAY)


def epsilon_greedy(action_value, epsilon, env):
	"""Apply epsilon greedy policy to given action value"""
	if random.uniform(0, 1) <= epsilon:
		return env.action_space.sample()  # random action
	else:
		return np.argmax(action_value)  # best action


##------------------------------------------------------------
##------------------ Tensorflow functions --------------------
def conv2D(x, W, b, stride=2):
    """Apply 2D conv + ReLU"""
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    conv = tf.nn.relu(tf.nn.bias_add(conv, b))
    return conv


def main_network(state, action, td_target, num_actions, alpha=0.001):
	"""Build TF computation graph"""
	weights = {
		'conv1': tf.Variable(tf.truncated_normal([6, 6, 4, 16], 0, 0.01)),
		'conv2': tf.Variable(tf.truncated_normal([4, 4, 16, 32], 0, 0.01)),
		'hidden': tf.Variable(tf.truncated_normal([7*7*32, 256], 0, 0.01)),
		'output': tf.Variable(tf.truncated_normal([256, num_actions], 0, 0.01))
	}
	biases = {
		'conv1': tf.Variable(tf.constant(0.1, shape=[16])),
		'conv2': tf.Variable(tf.constant(0.1, shape=[32])),
		'hidden': tf.Variable(tf.constant(0.1, shape=[256])),
		'output': tf.Variable(tf.constant(0.1, shape=[num_actions]))
	}

	# Convolutional layers (with ReLU)
	conv1 = conv2D(state, weights['conv1'], biases['conv1'], stride=2)
	conv2 = conv2D(conv1, weights['conv2'], biases['conv2'], stride=2)
	conv2_flat = tf.reshape(conv2,  [-1, 7*7*32])

	# Fully connected layers
	fc = tf.nn.relu(tf.matmul(conv2_flat, weights['hidden']) + biases['hidden'])
	q = tf.matmul(fc, weights['output']) + biases['output']

	# Calculate loss
	qa = tf.reshape(tf.gather_nd(q, action), [-1, 1])
	loss = tf.reduce_mean(tf.square(td_target - qa)/2)
	opt = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)
	return opt, q, loss


def target_network(state, reward, is_terminal, num_actions, gamma=0.99):
	"""Build TF computation graph"""
	weights_target = {
		'conv1': tf.Variable(tf.truncated_normal([6, 6, 4, 16], 0, 0.01)),
		'conv2': tf.Variable(tf.truncated_normal([4, 4, 16, 32], 0, 0.01)),
		'hidden': tf.Variable(tf.truncated_normal([7*7*32, 256], 0, 0.01)),
		'output': tf.Variable(tf.truncated_normal([256, num_actions], 0, 0.01))
	}
	biases_target = {
		'conv1': tf.Variable(tf.constant(0.1, shape=[16])),
		'conv2': tf.Variable(tf.constant(0.1, shape=[32])),
		'hidden': tf.Variable(tf.constant(0.1, shape=[256])),
		'output': tf.Variable(tf.constant(0.1, shape=[num_actions]))
	}

	# Convolutional layers
	conv1 = conv2D(state, weights_target['conv1'], biases_target['conv1'], stride=2)
	conv2 = conv2D(conv1, weights_target['conv2'], biases_target['conv2'], stride=2)
	conv2_flat = tf.reshape(conv2,  [-1, 7*7*32])

	# Fully connected layers
	fc = tf.nn.relu(tf.matmul(conv2_flat, weights_target['hidden']) + biases_target['hidden'])
	q_prime = tf.matmul(fc, weights_target['output']) + biases_target['output']

	q_prime_max = tf.reshape(tf.reduce_max(q_prime, reduction_indices=[1]), [-1, 1])
	target = reward + gamma * tf.multiply(is_terminal, q_prime_max)
	return target

def update_target_net(tf_vars, sess):
	"""Copy network weights from main net to target net"""
	# Build list of copy operations
	num_vars = int(len(tf_vars)/2)  # 4 in each net (2 weights, 2 biases)
	op_list = []
	for idx, var in enumerate(tf_vars[0:num_vars]):
		op_list.append(tf_vars[idx+num_vars].assign(var.value()))

	# Run the TF operations to copy variables
	for op in op_list:
		sess.run(op)

def init_tf_vars(num_actions):
	"""Initialize tensorflow variables"""
	state = tf.placeholder("float", [None, 28, 28, 4])
	action = tf.placeholder(tf.int32, [None, 2])
	td_target = tf.placeholder("float", [None, 1])
	reward = tf.placeholder("float", [None, 1])
	is_terminal = tf.placeholder("float", [None, 1])
	return state, action, td_target, reward, is_terminal


##---------------------------------------------------------------
##-------------------- Buffer -----------------------------------
class Buffer(object):
	def __init__(self, batch_size, max_buffer_size):
		self.min_size = batch_size
		self.max_size = max_buffer_size
		self.buffer = []

	def update(self, s, a, r, s_prime, terminal):
		"""Add new observations to buffer and truncate if too big"""
		if len(self.buffer) >= self.max_size:
			self.buffer = self.buffer[1:]
		self.buffer.append((s, a, r, s_prime, terminal))

	def check_size(self):
		"""Check if buffer is large enough to run a batch"""
		if len(self.buffer) > self.min_size:
			return True
		return False

	def random_sample(self, state, action, reward, is_terminal):
		"""Random sample from buffer"""
		sample = np.reshape(np.array(random.sample(self.buffer, self.min_size)), [self.min_size, 5])
		act = sample[:,1].reshape(-1, 1)
		act = np.append(np.arange(len(act)).reshape(-1, 1), act, axis=1)

		sample_dict_opt = {
			state: np.stack(sample[:,0], axis=0),
			action: act
		}

		sample_dict_target = {
			state: np.stack(sample[:,3], axis=0),
			reward: np.stack(sample[:,2], axis=0).reshape(-1, 1), 
			is_terminal: np.stack(sample[:,4], axis=0).reshape(-1, 1)
		}
		return sample_dict_opt, sample_dict_target


##---------------------------------------------------
##---------------------------------------------------
def main():
	# Training parameters
	max_steps = 1000000
	test_freq = 50000  # test every test_freq steps
	test_episodes = 10
	max_eps_length = 10000000
	gamma = 0.99

	# Hyperparameters
	max_buffer_size = 200000
	batch_size = 32
	alpha = 0.00001
	update_freq = 5000  # copy weights to target network every update_freq steps
	epsilon = 0.01


	# Variables to store results/counts
	mean_frame_count_list, mean_score_list, mean_opp_score_list, loss_list, mean_cum_rewards = [], [], [], [], []
	step_counter = 0
	game = 'MsPacman-v3'
	# game = 'Pong-v3' # 'MsPacman-v3' # 'Boxing-v3'


	##--------------------- Intialize gym environments, TF graphs and buffer
	env = gym.make(game)  	   # for Q-learning
	test_env = gym.make(game)  # for evaluation
	num_actions = env.action_space.n

	state, action, td_target, reward, is_terminal = init_tf_vars(num_actions)
	target = target_network(state, reward, is_terminal, num_actions, gamma=gamma)
	opt, q, loss = main_network(state, action, td_target, num_actions, alpha=alpha)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
	exp_buffer = Buffer(batch_size, max_buffer_size)  # initialize buffer
	t0 = time.time()
	##-----------------------------------------------------------------


	##-------------------- Training and evaluation -------------------------
	print("#-------------- Running OpenAI gym environment...")
	print("#-------------- Running %d steps of %s..." % (max_steps, game))
	with tf.Session() as sess:
		sess.run(init)

		while step_counter <= max_steps:
			env, s, obs = get_init_state(env)  # get initial state (4 frames), this also calls env.reset
			end = False

			##------------------ Run a single episode and perform Q update
			for t in range(max_eps_length):
				step_counter += 1

				# Take action, get state and reward
				q_now = sess.run(q, feed_dict={state: s.reshape(-1, 28, 28, 4)})
				# print(q_now)
				a = epsilon_greedy(q_now, epsilon, env)
				next_obs, r, done, info = env.step(a)
				r = clip_rewards(r)

				# Add to state (4 frames) and get is_terminal flag (0 if terminal, 1 otherwise)
				s_prime = update_state(s, obs)
				if done:
					terminal = 0
					end = True
				else:
					terminal = 1

				# Update buffer and perform mini-batch Q update if enough examples
				exp_buffer.update(s, a, r, s_prime, terminal)
				if exp_buffer.check_size():
					opt_dict, targ_dict = exp_buffer.random_sample(state, action, reward, is_terminal)
					targ = sess.run(target, feed_dict=targ_dict)
					opt_dict[td_target] = targ
					_, l = sess.run([opt, loss], feed_dict=opt_dict)
					loss_list.append(l)
				
				# Update target network parameters
				if step_counter % update_freq == 0:
					print("\n#------- After %d of %d steps, copying and saving neural net weights..." % (step_counter, max_steps))
					print("#------- Total time elapsed = %s\n" % str(time.time()-t0))
					train_vars = tf.trainable_variables()  # first half are main net, second half are target net
					update_target_net(train_vars, sess)
					save_path = saver.save(sess, '../../models/part2/'+game+'/tf_model')

				##--------------------------------------------------------------------------------------
				##----------------- Evaluate on 100 episodes every 50,000 steps (using test environment)
				if step_counter % test_freq == 0:
					score_list, opp_score_list, frame_count_list, cum_rewards_list = [], [], [], []

					print("\n#------ Running %d episodes for testing after %d of %d steps..." % (test_episodes, step_counter, max_steps))
					print("#------- Total time elapsed = %s\n" % str(time.time()-t0))
					for episode in range(test_episodes):
						progress_bar(episode, test_episodes)
						score = opp_score = frame_count = cum_reward = 0
						test_env, s_test, test_obs = get_init_state(test_env)

						for test_t in range(max_eps_length):
							q_now = sess.run(q, feed_dict={state: s_test.reshape(1, 28, 28, 4)})
							a = np.argmax(q_now)
							next_test_obs, test_r, done, info = test_env.step(a)
							cum_reward += clip_rewards(test_r) * gamma**test_t

							if test_r > 0:
								score += test_r
							else:
								opp_score -= test_r

							s_prime_test = update_state(s_test, test_obs)

							if done:
								score_list.append(score)
								opp_score_list.append(opp_score)
								frame_count_list.append(test_t+1)
								cum_rewards_list.append(cum_reward)
								break
							s_test = s_prime_test
							test_obs = next_test_obs

					## Update results lists
					mean_score_list.append(np.array(np.mean(score_list)))
					mean_opp_score_list.append(np.array(np.mean(opp_score_list)))
					mean_frame_count_list.append(np.array(np.mean(frame_count_list)))
					mean_cum_rewards.append(np.array(np.mean(cum_rewards_list)))

					# Print results
					results_tuple = (np.mean(np.array(score_list)), np.std(np.array(score_list)), 
									 np.mean(np.array(opp_score_list)), np.std(np.array(opp_score_list)), 
									 np.mean(np.array(frame_count_list)), np.std(np.array(frame_count_list)),
									 np.mean(np.array(cum_rewards_list)), np.std(np.array(cum_rewards_list)))
					print("\nMean score = %f, stdev score = %f, mean opp score = %f, stdev opp score = %f, mean fc = %f, stdev fc = %f, mean reward = %f, stdev reward = %f" % results_tuple)

					# Save performance and loss results
					save_results([range(0, step_counter, test_freq), mean_score_list, mean_opp_score_list, mean_frame_count_list, mean_cum_rewards],
						['step', 'score', 'opp_score', 'frame_count', 'cum_rewards'], 'results/atari_performance.csv')
					save_results([range(batch_size, step_counter), loss_list], ['step', 'loss'], 'results/atari_loss.csv')
				##--------------------------------------------------------------------------------------
				##--------------------------------------------------------------------------------------

				# Either end or continue from next state
				if end:
					break
				s = s_prime
				obs = next_obs

		sess.close()


if __name__ == "__main__":
	random.seed(0)
	main()
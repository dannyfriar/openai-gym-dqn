# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import math
import random
import time
import pdb
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import argparse
from collections import OrderedDict

##---------- Tensorflow graphs
def main_network(state, action, td_target, alpha=0.1, n_hidden=100):
	"""Build TF computation graph"""
	weights = {
	'hidden': tf.Variable(tf.random_uniform([4, n_hidden], 0, 0.01)),
	'output': tf.Variable(tf.random_uniform([n_hidden, 2], 0, 0.01))
	}
	biases = {
	'hidden': tf.Variable(tf.random_uniform([n_hidden], 0, 0.01)),
	'output': tf.Variable(tf.random_uniform([2], 0, 0.01))
	}

	hidden = tf.nn.relu(tf.matmul(state, weights['hidden']) + biases['hidden'])
	q = tf.matmul(hidden, weights['output']) + biases['output']

	# Calculate loss
	qa = tf.reshape(tf.gather_nd(q, action), [-1, 1])
	loss = tf.reduce_mean(tf.square(td_target - qa)/2)
	opt = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)
	return opt, q, loss

def target_network(state, reward, eps, rand_state, is_terminal, n_hidden=100, gamma=0.99):
	"""Build network to compute TD target"""
	weights_target = {
	'hidden': tf.Variable(tf.random_uniform([4, n_hidden], 0, 0.01)),
	'output': tf.Variable(tf.random_uniform([n_hidden, 2], 0, 0.01))
	}
	biases_target = {
	'hidden': tf.Variable(tf.random_uniform([n_hidden], 0, 0.01)),
	'output': tf.Variable(tf.random_uniform([2], 0, 0.01))
	}

	hidden = tf.nn.relu(tf.matmul(state, weights_target['hidden']) + biases_target['hidden'])
	q = tf.matmul(hidden, weights_target['output']) + biases_target['output']

	q_max = tf.reshape(tf.reduce_max(q, reduction_indices=[1]), [-1, 1])
	q_max = tf.multiply(q_max, eps) + tf.multiply(rand_state, 1-eps)
	target = reward + gamma * tf.multiply(is_terminal, q_max)
	return target

def init_vars():
	state = tf.placeholder("float", [None, 4])
	action = tf.placeholder(tf.int32, [None, 2])
	td_target = tf.placeholder("float", [None, 1])
	reward = tf.placeholder("float", [None, 1])
	eps = tf.placeholder("float", [None, 1])
	rand_state = tf.placeholder("float", [None, 1])
	is_terminal = tf.placeholder("float", [None, 1])
	return state, action, td_target, reward, eps, rand_state, is_terminal

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


##-------- Helper functions
def epsilon_greedy(action_value, epsilon, env):
	"""Apply epsilon greedy policy to given action value"""
	if random.uniform(0, 1) <= epsilon:
		return env.action_space.sample()  # random action
	else:
		return np.argmax(action_value)  # best action


##------- Buffer
class Buffer(object):
	def __init__(self, min_buffer_size, max_buffer_size, epsilon):
		self.min_size = min_buffer_size
		self.max_size = max_buffer_size
		self.epsilon = epsilon
		self.buffer = []

	def update(self, s, a, r, s_prime, term):
		"""Add new observations to buffer and truncate if too big"""
		if len(self.buffer) >= self.max_size:
			self.buffer = self.buffer[1:]
		self.buffer.append((np.reshape(s, (-1, 4)), a, r, np.reshape(s_prime, (-1, 4)), term))

	def check_size(self):
		"""Check if buffer is large enough to run a batch"""
		if len(self.buffer) > self.min_size:
			return True
		return False

	def random_sample(self, state, action, reward, eps, rand_state, is_terminal):
		"""Random sample from buffer"""
		sample = np.reshape(np.array(random.sample(self.buffer, self.min_size)), [self.min_size, 5])
		act = sample[:,1].reshape(-1,1)
		act = np.append(np.arange(len(act)).reshape(-1, 1), act, axis=1)

		sample_dict_opt = {
			state: np.stack(sample[:,0], axis=0).reshape(-1,4),
			action: act
		}

		sample_dict_target = {
			state: np.stack(sample[:,3], axis=0).reshape(-1,4),
			reward: np.stack(sample[:,2], axis=0).reshape(-1, 1),
			eps: np.random.binomial(01, 1-self.epsilon, self.min_size).reshape(-1, 1),
			rand_state: np.round(np.random.binomial(1, 0.5, self.min_size)).reshape(-1, 1),
			is_terminal: np.stack(sample[:,4], axis=0).reshape(-1, 1),
		}
		return sample_dict_opt, sample_dict_target


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run', default='')
	args = parser.parse_args()
	if args.run == "train":
		print("Retraining model...")
		n_episodes = 2000
		max_eps_length = 300
		reload_model = False
		save_data = True
	else:
		print("Testing model")
		n_episodes = 1
		max_eps_length = 0
		reload_model = True
		save_data = False

	# Training parameters
	print_iters = 200
	test_steps = 50
	test_freq = 20
	avg_steps = 1
	episodes_counter = 0

	min_buffer_size = 128
	max_buffer_size = 1000
	gamma = 0.99
	alpha = 0.0001
	epsilon = 0.05

	##--------------------- Initialize environment and TF graph
	env = gym.make('CartPole-v0')
	mean_length_list, mean_return_list, loss_list = [], [], []

	state, action, td_target, reward, eps, rand_state, is_terminal = init_vars()
	opt, q, loss = main_network(state, action, td_target, alpha=alpha)
	target = target_network(state, reward, eps, rand_state, is_terminal)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
	exp_buffer = Buffer(min_buffer_size, max_buffer_size, epsilon)
	copy_counter = 0

	with tf.Session() as sess:
		sess.run(init)

		for episode_i in range(n_episodes):
			t0 = time.time()
			end = False
			episode_return = 0
			copy_counter += 1
			s = env.reset()  # initial state
			return_list, length_list = [], []
			l_total = 0

			if reload_model:
				print("Reloading model...")
				folder = '../../models/part1/sarsa'
				saver = tf.train.import_meta_graph(folder+'/tf_model.meta')
				saver.restore(sess, tf.train.latest_checkpoint(folder+'/'))
				all_vars = tf.get_collection('vars')

			#------------- Perform updates and sample experiences (epsilon-greedy)
			for t in range(max_eps_length):
				q_now = sess.run(q, feed_dict={state: s.reshape(1, 4)})
				a = epsilon_greedy(q_now, epsilon, env)
				s_prime, _, done, info = env.step(a)
				if done:
					r = -1
					end = True
					term = 0
				else:
					r = 0
					term = 1

				# Update buffer and perform mini-batch Q update if enough examples
				exp_buffer.update(s, a, r, s_prime, term)
				if exp_buffer.check_size():
					opt_dict, targ_dict = exp_buffer.random_sample(state, action, reward, eps, rand_state, is_terminal)
					targ = sess.run(target, feed_dict=targ_dict)
					opt_dict[td_target] = targ
					_, l = sess.run([opt, loss], feed_dict=opt_dict)
					l_total += l

				# Update target network parameters
				if copy_counter == 5:
					train_vars = tf.trainable_variables()  # 1st 4 are main net, 2nd 4 are target net
					update_target_net(train_vars, sess)
					copy_counter = 0

				# Either end or continue from next state
				if end:
					if episode_i % test_freq == 0:
						loss_list.append(l_total/t)
					break
				s = s_prime
				#----------------------------------------------------------------------------

			# --------- Evaluate performance over test_steps episodes (using greedy policy)
			if episode_i % test_freq == 0:
				for i in range(test_steps):
					s = env.reset()
					for t in range(300):
						q_now = sess.run(q, feed_dict={state: s.reshape(1, 4)})
						a = np.argmax(q_now)
						s_prime, _, done, info = env.step(a)
						if done:
							episode_length = t + 1
							episode_return = -1 * gamma ** t
							length_list.append(episode_length)
							return_list.append(episode_return)
							break
						s = s_prime

				mean_length = np.mean(np.array(length_list))
				mean_return = np.mean(np.array(return_list))
				std_length = np.std(np.array(length_list))
				std_return = np.std(np.array(return_list))
				mean_length_list.append(mean_length)
				mean_return_list.append(mean_return)
				# save_path = saver.save(sess, '../../models/part1/sarsa/tf_model')
			# --------------------------------------------------------------------------

			if episode_i % print_iters == 0:
				print_tuple = (test_steps, mean_length, mean_return, std_length, std_return)
				print("#---After %d test episodes: Mean length = %f, mean return = %f, sd length = %f, sd return = %f" % print_tuple)
			episodes_counter += 1

		sess.close()

	# Save final results to CSV file
	if save_data:
		output_data = OrderedDict()
		output_data['episode'] = range(0, n_episodes, test_freq)
		output_data['length'] = mean_length_list
		output_data['return'] = mean_return_list
		output_data['loss'] = loss_list
		df = pd.DataFrame.from_dict(output_data)
		df.to_csv('results/sarsa_nn_results.csv', index=False)


if __name__ == "__main__":
	main()
from __future__ import division
from __future__ import print_function
import sys
import math
import random
import numpy as np
import pandas as pd
import gym

def progress_bar(value, endvalue, bar_length=20):
    """Print progress bar to the console"""
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def clip_rewards(reward):
	"""Clip rewards and increment score"""
	if reward > 0:
		return 1
	elif reward < 0:
		return -1
	else:
		return 0

def main():
	# Training parameters
	n_episodes = 100
	print_iters = 100
	max_eps_length = 100000
	gamma = 0.99
	save = True

	##--------------------- Run gym environment
	# game = 'Pong-v3'
	# game = 'MsPacman-v3'
	# game = 'Boxing-v3'
	# games = ['Pong-v3', 'MsPacman-v3', 'Boxing-v3']
	games = ['MsPacman-v3']

	print("#-------------- Running OpenAI gym environment...")
	for game in games:
		print("Running %d episodes of %s" % (n_episodes, game))
		env = gym.make(game)
		frame_count_list = []
		cum_rewards_list = []
		score_list = []
		opp_score_list = []
		
		for i_episode in range(n_episodes):
			progress_bar(i_episode, n_episodes)
			score = 0
			opp_score = 0
			cum_reward = 0
			frame_count = 0
			observation = env.reset()

			for t in range(max_eps_length):
				# env.render()
				action = env.action_space.sample()
				observation, reward, done, info = env.step(action)

				# Compute score (ignore opponent score for Ms Pacman) and clip reward
				if reward > 0:
					score += reward
				elif reward < 0:
					opp_score -= reward
				reward = clip_rewards(reward)
				cum_reward += reward

				if done:
					frame_count = t + 1
					break

			score_list.append(score)
			opp_score_list.append(opp_score)
			cum_rewards_list.append(cum_reward)
			frame_count_list.append(frame_count)

		# print(score_list)
		# print(opp_score_list)
		# print(cum_rewards_list)
		# print(frame_count_list)

		# Mean and stdev of scores
		results_tuple = (np.mean(np.array(score_list)), np.std(np.array(score_list)), 
			np.mean(np.array(opp_score_list)), np.std(np.array(opp_score_list)),
			np.mean(np.array(frame_count_list)), np.std(np.array(frame_count_list)))
		print("\nMean score = %f, stdev score = %f, mean opp score = %f, stdev opp score = %f, mean fc = %f, stdev fc = %f" % results_tuple)
		

if __name__ == "__main__":
	main()
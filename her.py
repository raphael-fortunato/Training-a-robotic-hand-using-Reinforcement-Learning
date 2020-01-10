import gym
from collections import deque
import numpy as np
import random
import pdb
from inspect import currentframe, getframeinfo
frameinfo = getframeinfo(currentframe())

from Prioritized_Experience_sampler import prioritized_sampler

class Buffer:
	def __init__(self, memory_size, her=True ,reward_func=None, per=True):
		self.her= her
		self.reward_function = reward_func
		self.buffer = deque(maxlen=memory_size)
		
		self.per = True
		if self.per:
			self.per_sampler = prioritized_sampler(metric='distance')
		
	def HERFutureBatch(self, batch):
		final_batch = []

		#Collect the goal of the each experience
		for i in range(len(batch)):
			substitute_goal = batch[i][0]["achieved_goal"]

			# Set the goal to that subsitute goal for the next 4 experiences (if possible)
			for x in range(4):
				if i + x < len(batch):
					final_batch.append(self.ChangeGoal(batch[i+x], substitute_goal))
		return final_batch
	
	#Changes the goal and reward for the given experience to the given goal
	def ChangeGoal(self, experience, goal):
		experience[0]["desired_goal"] =  goal
		experience[2][0] = self.reward_function(goal,  goal, experience[5])
		return experience
		
	def Sampler(self, batch_size, per_percentage):

		if self.per:
			# Creates a batch according to the preferred metric ("distance" or "impact") and the preferred distribution (number between 0 and 1)
			batch =  self.per_sampler.create_sample(self.buffer, batch_size, per_percentage )
		else:
			batch = random.sample(self.buffer, batch_size)

		# Extracts each aspect of the experiences
		observation = [data[0]['observation'] for data in batch]
		desired_goal = [data[0]['desired_goal'] for data in batch]
		reward = [data[2] for data in batch]
		action = [data[1] for data in batch]
		done = [data[3] for data in batch]
		observationt1 = [data[4]['observation'] for data in batch]
		desired_goalt1 = [data[4]['desired_goal'] for data in batch]

		# Return the necessary extractions
		return np.concatenate([observation, desired_goal], axis=1), action, reward, done, np.concatenate([observationt1, desired_goalt1], axis=1)

	def append(self, experience):
		self.buffer.append(experience)
		

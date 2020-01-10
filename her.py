import gym
from collections import deque
import numpy as np
import random
import pdb

from Prioritized_Experience_sampler import prioritized_sampler

class Buffer:
    def __init__(self, memory_size,per=True ,her=True ,reward_func=None):
        self.her= her
        self. reward_function = reward_func
        self.buffer = deque(maxlen=memory_size)
        self.per = per
        if self.per:
            self.per_sampler = prioritized_sampler(metric='distance')

    def sampler(self, batch_size, her_size, per_percentage):
        if self.per:
            #Creates a batch according to the preferred metric ("distance" or "impact") and the preferred distribution (number between 0 and 1)
            batch =  self.per_sampler.create_sample(self.buffer, batch_size, per_percentage )
        else:
            batch = random.sample(self.buffer, batch_size)
        
        length = int(her_size * batch_size)

        observation = [data[0]['observation'] for data in batch]
        achieved_goal = [data[0]['achieved_goal'] for data in batch]
        desired_goal = [data[0]['desired_goal'] for data in batch]
        reward = [data[2] for data in batch]
        action = [data[1] for data in batch]
        done = [data[3] for data in batch]
        observationt1 = [data[4]['observation'] for data in batch]
        desired_goalt1 = [data[4]['desired_goal'] for data in batch]
        info = [data[5] for data in batch]

        if self.her:
            substitute_goal = achieved_goal[:length]
            new_reward = self.reward_function(np.array(achieved_goal[:length]), np.array(substitute_goal), np.array(info[:length]))
            reward[:length] = new_reward
            desired_goal = substitute_goal + desired_goal[length:]

        return np.concatenate([observation, desired_goal], axis=1), action, reward, done, np.concatenate([observationt1, desired_goalt1], axis=1)


    def append(self, experience):
        self.buffer.append(experience)

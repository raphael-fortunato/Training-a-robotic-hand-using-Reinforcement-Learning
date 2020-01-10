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
        for i in range(len(batch)):
            for x in range(4):
                if i + x < len(batch):
                    final_batch.append(self.ChangeGoal(batch[i+x]))
        
        return final_batch
    
    def ChangeGoal(self, experience):
        substitute_goal = experience[0]["achieved_goal"]
        experience[0]["desired_goal"] =  substitute_goal
        experience[2][0] = self.reward_function(substitute_goal,  substitute_goal, experience[5])
        
    def Sampler(self, batch_size, per_percentage):
        if self.per:
            #Creates a batch according to the preferred metric ("distance" or "impact") and the preferred distribution (number between 0 and 1)
            batch =  list(self.per_sampler.create_sample(self.buffer, batch_size, per_percentage ))
        else:
            batch = list(random.sampler(self.buffer, batch_size))
        
        return batch

    def append(self, experience):
        self.buffer.append(experience)
        

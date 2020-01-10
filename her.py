import gym
from collections import deque
import numpy as np
import random
import pdb
from Prioritized_Experience_sampler import prioritized_sampler

class Buffer:
    def __init__(self, memory_size,num_threads,per=True ,her=True ,reward_func=None):
        self.her= her
        self.reward_function = reward_func
        self.buffer = deque(maxlen=memory_size)
        self.per = per
        self.n_threads = num_threads
        if self.per:
            self.per_sampler = prioritized_sampler(metric='distance')
        
    def HERFutureBatch(self, batch):
        for thread in range(self.n_threads):
            data = [exp for exp in batch if thread == exp[-1]]
            final_batch = []
            for i in range(len(data)):
                for x in range(4):
                    if i + x < len(data):
                        final_batch.append(self.ChangeGoal(data[i+x]))
        
        return final_batch
    
    def ChangeGoal(self, experience):
        experience = list(experience)
        substitute_goal = experience[0]["achieved_goal"]
        experience[0]["desired_goal"] =  substitute_goal
        return tuple(experience)
        
    def Sampler(self, batch_size, per_percentage):
        if self.per:
            #Creates a batch according to the preferred metric ("distance" or "impact") and the preferred distribution (number between 0 and 1)
            batch =  list(self.per_sampler.create_sample(self.buffer, batch_size, per_percentage ))
        else:
            batch = random.sample(self.buffer, batch_size)

        observation = [data[0]['observation'] for data in batch]
        achieved_goal = [data[0]['achieved_goal'] for data in batch]
        desired_goal = [data[0]['desired_goal'] for data in batch]
        reward = [data[2] for data in batch]
        action = [data[1] for data in batch]
        done = [data[3] for data in batch]
        observationt1 = [data[4]['observation'] for data in batch]
        desired_goalt1 = [data[4]['desired_goal'] for data in batch]
        info = [data[5] for data in batch]

        return np.concatenate([observation, desired_goal], axis=1), action, reward, done, np.concatenate([observationt1, desired_goalt1], axis=1)

    def append(self, experience):
        self.buffer.append(experience)
        

    def concat(self, data):
        self.buffer.extendleft(data)

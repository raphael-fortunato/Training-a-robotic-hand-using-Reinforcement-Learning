import numpy as np
import random

class prioritized_sampler:
    def __init__(self):
        pass

    # Calculates the Euclidean distance between two vectors
    def Distance(self, achieved_goal, desired_goal):
        return np.sum(np.power(desired_goal - achieved_goal, 2))

    # Calculates how the hand has moved between states
    def Impact(self, currentState, nextState):
        return np.sum(np.power(nextState - currentState, 2))

    # Return methods for the sorting algorithm
    def getDistance(self, item):
        return self.Distance(item[0]["achieved_goal"],item[0]["desired_goal"])

    def getImpact(self, item):
        return self.Impact(item[0]["observation"],item[4]["observation"])

    #Samples the memory according to the given distribution and according to the given metric
    def create_sample(self, memory, sample_size, percentage, metric):

        # Make a list and then sort it according to the given metric
        newMemory = list(memory)
        if metric == "distance":
            newMemory = sorted(newMemory, key = self.getDistance)
        elif metric == "impact":
            newMemory = sorted(newMemory, key = self.getImpact)
        else:
            raise KeyError("Mode does not exist, only \"distance\" or \"impact\"")

        # Slice the sorted memory in half
        lowestHalf = newMemory[:int(len(newMemory)/2)]
        topHalf = newMemory[int(len(newMemory)/2):]

        # Get the samples from the top half and the lower half, according to the percentage that was given
        sample = list(random.sample(topHalf, int(round(sample_size*percentage, 0))))
        newsample = list(random.sample(lowestHalf, sample_size-int(round(sample_size*percentage, 0)))) 

        # Concatenate into 1 list and return. 
        return np.concatenate([sample, newsample], axis = 0)
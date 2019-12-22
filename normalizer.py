import numpy as np


class Normalizer:
    def __init__(self, minimum, maximum):
        self.range = [minimum,-5,0,5,maximum]

    def normalize(self, data):
        return np.interp(data, range, [-1,-.5,0, .5, 1] )



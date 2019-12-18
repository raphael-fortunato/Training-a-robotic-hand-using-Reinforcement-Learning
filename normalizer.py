import numpy as np


class Normalizer:
    def __init__(self, minimum, maximum):
        self.range = [minimum,-5,0,5,maximum]

    def normalize(self, data):
        return np.interp(data, range, [-1,-.5,0, .5, 1] )


# norm = Normalizer(-10,10 )
# a = np.random.randint(-10,10, size=(5,5))
# print(a)
# test = norm.normalize(a)
# print(test)
import gym
from multiprocessing import Process
from VectorEnv import VectorEnv

def Main():
    vecenv = VectorEnv(gym.make('BipedalWalker-v2))

if __name__ == '__main__':
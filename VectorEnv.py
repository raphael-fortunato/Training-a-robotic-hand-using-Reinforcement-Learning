import gym
import multiprocessing
from multiprocessing import Process, Queue, Pool
from Main import Agent
import numpy as np
import pdb
from library.stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def get_params(env):
    obs = env.reset()

    params = {'observation': obs.shape[0],
            'goal' : np.empty(0).shape[0],
            'action': env.action_space.shape[0],
            'max_action': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params



if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    env_make = tuple(lambda: gym.make('BipedalWalker-v2') for _ in range(2))
    test = SubprocVecEnv(env_make)
    pdb.set_trace()
    agent = Agent(env,get_params(env), n_episodes=2_000,save_path=None ,noise_eps=3., batch_size=2560 ,her=False, per=False ,screen=False, tensorboard =False)
    vecenv = VectorEnv(lambda : gym.make('BipedalWalker-v2'), agent)
    pdb.set_trace()

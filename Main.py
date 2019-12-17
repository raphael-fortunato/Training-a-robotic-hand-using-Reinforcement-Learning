import gym 
import pdb

import numpy as np
from collections import deque

import torch

from rl.memory import SequentialMemory

from models import Actor, Critic


class Agent:
    def __init__(self, env, env_params):
        self.env= env
        self.env_params = env_params
        self.hidden_neurons = 256
        # networks
        self.actor = Actor(self.env_params, self.hidden_neurons)
        self.critic = Critic(self.env_params, self.hidden_neurons)
        # target networks used to predict env actions with
        self.actor_target = Actor(self.env_params, self.hidden_neurons)
        self.critic_target = Critic(self.env_params, self.hidden_neurons)
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory = SequentialMemory(limit=1_000_000, window_length=1)





def get_params(env):
    obs = env.reset()
    params = {'observation': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'max_action': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params


env = gym.make('HandManipulateBlock-v0')
env_param = get_params(env)
agent = Agent(env,env_param)
env.close()

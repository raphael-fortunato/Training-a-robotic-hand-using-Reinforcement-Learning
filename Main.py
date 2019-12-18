import os
import gym 
import pdb

import numpy as np
import random
from collections import deque

import torch

from rl.memory import SequentialMemory

from models import Actor, Critic
from normalizer import Normalizer

class Agent:
    def __init__(self, env, env_params, n_episodes, noise_eps ,screen=False,save_path='models'):
        self.env= env
        self.env_params = env_params
        self.episodes = n_episodes
        self.hidden_neurons = 256
        self.noise_eps = noise_eps
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
        # create path to save the model
        self.model_path = (save_path + 'Actor', save_path + 'Critic')
        if not os.path.exists(self.model_path[0]):   
            os.mkdir(self.model_path[0])
        if not os.path.exists(self.model_path[1]):   
            os.mkdir(self.model_path[1])
        self.screen = screen


    def Action(self, step):
        with torch.no_grad():
            action = self.actor_target.forward(step).detach().cpu().numpy()
            action +=self.noise_eps * self.env_params['max_action'] * np.random.randn(*action.shape)
            action = np.clip(action, -self.env_params['max_action'], self.env_params['max_action'])
            # random actions...
            random_actions = np.random.uniform(low=-self.env_params['max_action'], high=self.env_params['max_action'], \
                                            size=self.env_params['action'])
            # choose if use the random actions
            # need to specify random.uniform
            action += np.random.binomial(1, random.uniform(0,1), 1)[0] * (random_actions - action)
        return action.squeeze()


    def Update(self):
        batch  = self.memory.sample(32)
        pass

    def Learn(self):
        for episode in range(self.episodes):
            state = env.reset()
            for t in range(self.env_params['max_timesteps']): 
                if self.screen:
                    env.render()
                #NORMALIZE EVERY INPUT 
                #pdb.set_trace()
                #norm = Normalizer(-10, 10)
                #test = norm.normalize(state)
                action = self.Action(state)

                nextstate, reward, done, info = env.step(action)
                self.memory.append(state, action, reward, done, nextstate)
                state = nextstate
                if done:
                    break



    def SaveModel(self):
        torch.save(self.actor.state_dict(), self.model_path[0])
        torch.save(self.critic.state_dict(), self.model_path[1])



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
agent = Agent(env,env_param, 3, 1., screen=True)
agent.Learn()
env.close()

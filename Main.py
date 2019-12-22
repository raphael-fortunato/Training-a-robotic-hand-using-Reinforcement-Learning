import os
import gym 
import pdb

import numpy as np
import random
from collections import deque

import torch
#from torch.utils import SummaryWritter


from models import Actor, Critic
from normalizer import Normalizer
from her import Buffer
#from library.stable_baselines.common.vec_env.base_vec_env.vec_normalize import  VecNormalize





class Agent:
    def __init__(self, env, env_params, n_episodes, noise_eps,batch_size=32, her_size=.5 ,gamma=.99 ,screen=False,save_path='models'):
        self.env= env
        self.env_params = env_params
        self.episodes = n_episodes
        self.hidden_neurons = 256
        self.noise_eps = noise_eps
        self.gamma = gamma
        self.batch_size = batch_size
        # networks
        self.actor = Actor(self.env_params, self.hidden_neurons).double()
        self.critic = Critic(self.env_params, self.hidden_neurons).double()
        # target networks used to predict env actions with
        self.actor_target = Actor(self.env_params, self.hidden_neurons).double()
        self.critic_target = Critic(self.env_params, self.hidden_neurons).double()
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory = deque(maxlen=1_000_000)
        # create path to save the model
        self.model_path = (save_path + 'Actor', save_path + 'Critic')
        if not os.path.exists(self.model_path[0]):   
            os.mkdir(self.model_path[0])
        if not os.path.exists(self.model_path[1]):   
            os.mkdir(self.model_path[1])
        self.screen = screen
        self.buffer = Buffer(1_000_000, her=True, reward_func=self.env.compute_reward, per=True)
        self.her_size = her_size
        #self.normalizer = vec_normalize()
        #self.tensorboard = SummaryWritter()



    def Action(self, step):
        
        with torch.no_grad():
            step = np.concatenate([step['observation'], step['desired_goal']])
            step = torch.tensor(step)
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
        state, a_batch, r_batch, d_batch, nextstate = self.buffer.sampler(self.batch_size, self.her_size, .8)

        a_batch = torch.tensor(a_batch,dtype=torch.double)
        r_batch = torch.tensor(r_batch,dtype=torch.double)
        d_batch = torch.tensor(d_batch,dtype=torch.double)
        state = torch.tensor(state, dtype=torch.double)
        nextstate = torch.tensor(nextstate, dtype=torch.double)

        if torch.cuda.is_available():
            a_batch = a_batch.cuda()
            r_batch = r_batch.cuda()
            d_batch = d_batch.cuda()
            state = state.cuda()
            nextstate = nextstate.cuda()
    
        with torch.no_grad():
            action_next = self.actor_target.forward(state)
            q_next = self.critic_target.forward(nextstate,action_next)
            q_next = q_next.detach()

        q_target = r_batch + (self.gamma *(1 - d_batch) * q_next)
        q_target = q_target.detach()
        q_prime = self.critic.forward(state, a_batch)
        critic_loss = (q_target - q_prime).pow(2).mean()

        action = self.actor.forward(state)
        actor_loss = -self.critic.forward(state, a_batch).mean()
        actor_loss += (action / self.env_params['max_action']).pow(2).mean()

        print(f"actor loss = {actor_loss}, critic loss = {critic_loss}")

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return actor_loss, critic_loss

        # COPY WEIGHTS INTO TARGET NETWORKS EVERY # OF STEPS
        
    def SaveModel(self):
        torch.save(self.actor.state_dict(), self.model_path[0])
        torch.save(self.critic.state_dict(), self.model_path[1])   

    def Explore(self):
        succes_rate = []
        for episode in range(self.episodes):
            state = env.reset()
            for t in range(self.env_params['max_timesteps']): 

                if self.screen:
                    env.render()

                #NORMALIZE EVERY INPUT 
                action = self.Action(state)
                nextstate, reward, done, info = env.step(action)
                self.buffer.append((state, action, reward, done, nextstate, info))
                state = nextstate
               
                if reward ==1:
                    succes_rate.append(True)
                if done:
                    if not episode % 100:
                        pass
                        #self.tensorboard.add_scalar('Loss', loss)
                    succes_rate.append(False)
                    print(f"Episode: {episode}of{self.episodes}, succes_rate:{np.sum(succes_rate)/len(succes_rate)}")
                    break
            if not episode % 5 and episode != 0:  
                print('Training Networks')       
                self.Update()


def get_params(env):
    obs = env.reset()
    params = {'observation': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'max_action': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

#env = gym.make('MountainCarContinuous-v0')
env = gym.make("HandManipulateBlock-v0")
env_param = get_params(env)
agent = Agent(env,env_param, 30, 1., screen=False)
agent.Explore()
env.close()

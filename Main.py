import os
import gym 
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import pdb

import numpy as np
import random
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from normalizer import Normalizer
from models import Actor, Critic
from her import Buffer
from CustomTensorBoard import ModifiedTensorBoard


class Agent:
    def __init__(self, env, env_params, n_episodes,noise_eps, random_eps=.3,batch_size=62, her_size=.5, \
                gamma=.99, per=True, her=True ,screen=False, agent_name='ddpg',save_path='models', record_episode = [0,200, 500, 1000] ,aggregate_stats_every=100):
        self.env= env
        self.env_params = env_params
        self.episodes = n_episodes
        self.hidden_neurons = 256
        self.noise_eps = noise_eps
        self.random_eps = random_eps
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
        self.buffer = Buffer(1_000_000, per=per ,her=her,reward_func=self.env.compute_reward,)
        self.her_size = her_size
        self.norm = Normalizer(self.env_params, self.gamma)
        self.writer = SummaryWriter(f"runs/{agent_name}")
        self.tensorboard = ModifiedTensorBoard(log_dir = f"logs")
        self.aggregate_stats_every =aggregate_stats_every
        self.record_episodes = record_episode


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
            action += np.random.binomial(1, self.random_eps, 1)[0] * (random_actions - action)
        return action.squeeze()

    def Update(self, episode):
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

        #writting to tensorboard for visualisation
        self.tensorboard.update_stats(ActorLoss=actor_loss/self.batch_size, CriticLoss=critic_loss/self.batch_size, episode=episode)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        # COPY WEIGHTS INTO TARGET NETWORKS EVERY # OF STEPS
    
        return actor_loss, critic_loss


    def record(self, episode):
        try:
            if not os.path.exists("videos"):
                os.mkdir('videos')
            recorder = VideoRecorder(self.env, path=f'videos\\episode-{episode}.mp4')
            done =False
            step = self.env.reset()
            step = self.norm.normalize_state(step)
            while not done:
                recorder.capture_frame()
                action = self.Action(step)
                nextstate,reward,done,info = self.env.step(action)
                nextstate = self.norm.normalize_state(nextstate)
                state = nextstate
            recorder.close()
        except Exception as e:
            print(e)

    def SaveModel(self):
        torch.save(self.actor.state_dict(), self.model_path[0])
        torch.save(self.critic.state_dict(), self.model_path[1])   

    def Explore(self):
        succes_rate = []
        ep_rewards = []
        significant_moves = []
        for episode in range(self.episodes):
            state = env.reset()
            state = self.norm.normalize_state(state)
            self.tensorboard.step = episode
            total_rewards = 0
            significance = 0
            for t in range(self.env_params['max_timesteps']): 
                if self.screen:
                    env.render()

                action = self.Action(state)
                nextstate, reward, done, info = env.step(action)

                total_rewards += reward
                if np.sum(nextstate['desired_goal'] - nextstate['achieved_goal']) == 0:
                    succes_rate.append(True)

                nextstate = self.norm.normalize_state(nextstate)
                reward = self.norm.normalize_reward(reward)
                self.buffer.append((state, action, reward, done, nextstate, info))

                significance += abs(np.sum(state['observation'] - nextstate['observation']))
                state = nextstate


                if done:
                    if episode in self.record_episodes:
                        self.record(episode)
                    succes_rate.append(False)
                    ep_rewards.append(total_rewards)
                    significant_moves.append(significance)
                    print(f"Episode: {episode}of{self.episodes}, succes_rate:{np.sum(succes_rate)/len(succes_rate)}")
                    if not episode % self.aggregate_stats_every:
                        average_reward = sum(ep_rewards[-self.aggregate_stats_every:])/len(ep_rewards[-self.aggregate_stats_every:])
                        min_reward = min(ep_rewards[-self.aggregate_stats_every:])
                        max_reward = max(ep_rewards[-self.aggregate_stats_every:])
                        succes = np.sum(succes_rate[-self.aggregate_stats_every:])/len(succes_rate[-self.aggregate_stats_every:])
                        sig = sum(significant_moves[-self.aggregate_stats_every:])/len(significant_moves[-self.aggregate_stats_every:])
                        self.tensorboard.update_stats(Succes_Rate=succes,reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, SignificantMove=sig)
                    break
            if not episode % 5 and episode != 0:  
                print('Training Networks')       
                self.Update(episode)




def get_params(env):
    obs = env.reset()
    params = {'observation': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'max_action': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params


env = gym.make("HandManipulateBlock-v0")
env_param = get_params(env)
agent = Agent(env,env_param, n_episodes=11, noise_eps=1., batch_size=256, screen=True)
agent.Explore()
env.close()

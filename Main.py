import os
import gym 
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import pdb
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from normalizer import Normalizer
from models import Actor, Critic
from her import Buffer
from CustomTensorBoard import ModifiedTensorBoard
import time


class Agent:
    def __init__(self, env, env_params, n_episodes,noise_eps,tau=.95, random_eps=.3,batch_size=256, her_size=.5, \
                gamma=.99, per=True, her=True ,screen=False,modelpath='models' ,savepath=None ,agent_name='ddpg',save_path='models', record_episode = [0 ,.1 , .15, .25, .5, .75] ,aggregate_stats_every=100):
        self.env= env
        self.env_params = env_params
        self.episodes = n_episodes
        self.hidden_neurons = 14
        self.noise_eps = noise_eps
        self.random_eps = random_eps
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        # networks
        if savepath == None:
            self.actor = Actor(self.env_params, self.hidden_neurons).double()
            self.critic = Critic(self.env_params, self.hidden_neurons).double()
        else:
            self.actor , self.critic = self.LoadModels(savepath[0], savepath[1])
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
        # create path to save the model
        self.savepath = save_path
        self.path = modelpath
        if not os.path.exists(self.path):   
            os.mkdir(self.path)
        self.screen = screen
        self.buffer = Buffer(1_000_000, per=per ,her=her,reward_func=self.env.compute_reward,)
        self.her_size = her_size
        self.norm = Normalizer(self.env_params, self.gamma)
        self.writer = SummaryWriter(f"runs/{agent_name}")
        self.tensorboard = ModifiedTensorBoard(log_dir = f"logs")
        self.aggregate_stats_every =aggregate_stats_every
        self.record_episodes = [int(eps *self.episodes) for eps in record_episode]


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
            action += np.random.binomial(1, self.random_eps, 1)[0] * (random_actions - action)
            self.noise_eps *= .99
        return action

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

        q_target = r_batch + self.gamma * q_next
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
        
        self.SoftUpdateTarget(self.critic, self.critic_target)
        self.SoftUpdateTarget(self.actor, self.actor_target)
    
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

    def SaveModels(self):
        torch.save(self.actor.state_dict(), self.savepath+ "\\Actor.pt")
        torch.save(self.critic.state_dict(), self.savepath+ "\\Critic.pt") 

    def LoadModels(self, actorpath, criticpath):
        actor = Actor(self.env_params, self.hidden_neurons)
        critic  = Critic(self.env_params, self.hidden_neurons)
        actor.load_state_dict(torch.load(actorpath))
        critic.load_state_dict(torch.load(criticpath))
        return actor, critic

    def SoftUpdateTarget(self, network, target):
        for target_param, param in zip(network.parameters(), target.parameters()):
             target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


    def Explore(self):
        first = True
        onlyfiles = []
        succes_rate = []
        ep_rewards = []
        significant_moves = []
        timeline = []
        iterator = tqdm(range(self.episodes+1), unit='episode')
        for episode in iterator:
            start = time.time()
            state = env.reset()
            #state = self.norm.normalize_state(state)
            self.tensorboard.step = episode
            total_rewards = 0
            significance = 0
            for t in range(self.env_params['max_timesteps']): 
                if self.screen:
                    env.render()

                action = self.Action(state)
                nextstate, reward, done, info = env.step(action)

                total_rewards += reward
                # if info['is_success']:
                #     succes_rate.append(True)

                # nextstate = self.norm.normalize_state(nextstate)
                # reward = self.norm.normalize_reward(reward)
                self.buffer.append((state, action, reward, done, nextstate, info))

                significance += abs(np.sum(state - nextstate))
                state = nextstate

                if done:
                    if episode in self.record_episodes:
                        self.record(episode)
                    # if not info['is_success']:
                    #     succes_rate.append(False)
                    ep_rewards.append(total_rewards)
                    significant_moves.append(significance)
                    end = time.time()
                    timeline.append(end-start)
                    iterator.set_postfix(average_reward = sum(ep_rewards)/len(ep_rewards), epsilon = self.noise_eps)
                    if not episode % self.aggregate_stats_every:
                        average_reward = sum(ep_rewards)/len(ep_rewards)
                        min_reward = min(ep_rewards)
                        max_reward = max(ep_rewards)
                        sig = sum(significant_moves)/len(significant_moves)
                        timing =sum(timeline)
                        self.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, SignificantMove=sig, elapsed_time=timing)
                        timeline.clear() , significant_moves.clear()

                    break
            if not episode % 10 and episode != 0:  
                iterator.write(f'Training Networks - {episode}')       
                self.Update(episode)
        self.SaveModels()


def get_params(env):
    obs = env.reset()

    params = {'observation': obs.shape[0],
            'goal' : np.empty(0).shape[0],
            'action': env.action_space.shape[0],
            'max_action': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params


#loadmodels = ('models//Actor.pt', 'models//Critic.pt')
#env = gym.make("HandManipulateBlock-v0")
env = gym.make('MountainCarContinuous-v0')

env_param = get_params(env)
agent = Agent(env,env_param, n_episodes=50, noise_eps=3., batch_size=256 ,screen=True)
agent.Explore()
env.close()

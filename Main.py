import os
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import pdb
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import torch
import torch.nn.functional as F
from normalizer import Normalizer
from models import Actor, Critic
from her import Buffer
from CustomTensorBoard import ModifiedTensorBoard
from OUnoise import OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec, Distance
from tensorboard import default
from tensorboard import program
import threading
import time
from copy import deepcopy
from library.stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import argparse




class Agent:
    def __init__(self, test_env ,env, env_params, n_episodes,n_threads ,tensorboard=True, random_eps=.3 ,noise_eps=.2, tau=0.05, batch_size=256, \
                gamma=.99, l2=1. ,per=True, her=True ,screen=False,modelpath='models' ,savepath=None, save_path='models',\
                record_episode = [0,.05 ,.1 , .15, .25,.35 ,.5, .75, 1.] ,aggregate_stats_every=100):
        self.evaluate_env = test_env
        self.env= env
        self.env_params = env_params

        self.episodes = n_episodes
        self.hidden_neurons = 256
        self.noise_eps = noise_eps
        self.random_eps = random_eps
        self.gamma = gamma
        self.tau = tau
        self.l2_norm = l2
        self.batch_size = batch_size
        self.param_noise = AdaptiveParamNoiseSpec()
        self.noise = OrnsteinUhlenbeckActionNoise(np.zeros((n_threads, self.env_params['action'])))

        # networks
        if savepath == None:
            self.actor = Actor(self.env_params, self.hidden_neurons).double()
            self.critic = Critic(self.env_params, self.hidden_neurons).double()
        else:
            self.actor , self.critic = self.LoadModels(savepath[0], savepath[1])
        # target networks used to predict env actions with
        self.actor_target = Actor(self.env_params, self.hidden_neurons).double()
        self.critic_target = Critic(self.env_params, self.hidden_neurons).double()
        self.actor_pertubated = Actor(self.env_params, self.hidden_neurons).double()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()
            self.actor_pertubated.cuda()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        # create path to save the model
        self.savepath = save_path
        self.path = modelpath
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.screen = screen
        self.her = her
        self.buffer = Buffer(1_000_000,num_threads =n_threads ,per=per ,her=her,reward_func=self.evaluate_env.compute_reward,)
        self.norm = Normalizer(self.env_params, self.gamma)
        self.tensorboard = ModifiedTensorBoard(log_dir = f"logs")
        self.aggregate_stats_every =aggregate_stats_every
        self.record_episodes = [int(eps *self.episodes) for eps in record_episode]
        if tensorboard:
            self.t = threading.Thread(target=self.LaunchTensorBoard, args=([]))
            self.t.start()

    def LaunchTensorBoard(self):
        os.system('tensorboard --logdir=' + 'logs'+ ' --host 0.0.0.0')


    def perturbate_actor(self):
        self.actor_pertubated.load_state_dict(self.actor_target.state_dict())
        params = self.actor_pertubated.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            if torch.cuda.is_available():
                param.copy_(param + torch.randn(param.shape, device='cuda') * param_noise.current_stddev)
            else:
                param.copy_(torch.randn(param.shape) * param_noise.current_stddev)
        action = self.actor_pertubated.forward(state).detach().cpu().numpy()
        self.param_noise.adapt(Distance(self.actor.forward(state).detach().cpu().numpy(), action))


    def Action(self, state, param_noise, batch= True):
        with torch.no_grad():
            if batch:
                state = np.concatenate([state['observation'], state['desired_goal']], axis=1)
            else:
                state = np.concatenate([state['observation'], state['desired_goal']])
            if param_noise is not None:
                action = self.actor_pertubated.forward(state).detach().cpu().numpy()
                self.param_noise.adapt(Distance(self.actor.forward(state).detach().cpu().numpy(), action))
                #random actions
                random_actions = np.random.uniform(low=-self.env_params['max_action'], high=self.env_params['max_action'], \
                                            size=self.env_params['action'])
                action += np.random.binomial(1, self.random_eps, 1)[0] * (random_actions - action)
                return action
            else:
                return self.actor.forward(state).detach().cpu().numpy()


    def Update(self, iteration):
        c_loss = []
        a_loss = []
        for i in range(iteration):
            state, a_batch, r_batch, d_batch, nextstate = self.buffer.Sampler(self.batch_size, .8)
            a_batch = torch.tensor(a_batch,dtype=torch.double)
            r_batch = torch.tensor(r_batch,dtype=torch.double)
            d_batch = torch.tensor(d_batch,dtype=torch.double)
            state = torch.tensor(state, dtype=torch.double)
            nextstate = torch.tensor(nextstate, dtype=torch.double)
            d_batch = 1 - d_batch

            if torch.cuda.is_available():
                a_batch = a_batch.cuda()
                r_batch = r_batch.cuda()
                d_batch = d_batch.cuda()
                state = state.cuda()
                nextstate = nextstate.cuda()

            with torch.no_grad():
                action_next = self.actor_target.forward(nextstate)
                q_next = self.critic_target.forward(nextstate,action_next)
                q_next = q_next.detach().squeeze()
                q_target = r_batch + (self.gamma * q_next *d_batch)
                q_target = q_target.detach()

            q_prime = self.critic.forward(state, a_batch)
            critic_loss = F.mse_loss(q_target.squeeze() , q_prime.squeeze())

            action = self.actor.forward(state)
            actor_loss = -self.critic.forward(state, action).mean()
            actor_loss += self.l2_norm * (action / self.env_params['max_action']).pow(2).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            a_loss.append(actor_loss.item()/ self.batch_size)
            c_loss.append(critic_loss.item()/ self.batch_size)

        self.SoftUpdateTarget(self.critic, self.critic_target)
        self.SoftUpdateTarget(self.actor, self.actor_target)
        
        return np.sum(a_loss)/ len(a_loss), np.sum(c_loss) / len(c_loss)


    def Explore(self, iteration):
        iterator = tqdm(range(self.episodes +1), unit='episode')
        for episode in iterator:
            temp_buffer = []
            state = self.env.reset()
            state = self.norm.normalize_state(state)
            self.tensorboard.step = episode
            for t in range(self.env_params['max_timesteps']): 

                if self.screen:
                    self.env.render()

                action = self.Action(state, param_noise=self.param_noise)
                nextstate, reward, done, info = self.env.step(action)

                nextstate = self.norm.normalize_state(nextstate)
                reward = self.norm.normalize_reward(reward)
                
                for id, (a,r,d,i) in enumerate(zip(action,reward,done,info)):
                    experience = ({'achieved_goal':state['achieved_goal'][id], 'desired_goal': state['desired_goal'][id], 'observation': state['observation'][id] },\
                     a, r, d, \
                     {'achieved_goal':nextstate['achieved_goal'][id], 'desired_goal': nextstate['desired_goal'][id], 'observation': nextstate['observation'][id] }, \
                     i,id )
                    temp_buffer.append(experience)
                state = nextstate
                if np.array(done).any() or t +1 == self.env_params['max_timesteps']:
                    if self.her:
                        her_batch = self.buffer.HERFutureBatch(deepcopy(temp_buffer))
                        self.buffer.concat(her_batch)
                    self.buffer.concat(deepcopy(temp_buffer))
                    if episode > 5:
                        self.perturbate_actor()
                        a_loss, c_loss = self.Update(iteration)
                        self.tensorboard.update_stats(ActorLoss=a_loss/self.batch_size, CriticLoss=c_loss)
                        iterator.set_postfix(Actor_loss = a_loss, Critic_loss=c_loss)
                    if not episode % self.aggregate_stats_every:
                        self.Evaluate()
                    if episode in self.record_episodes:
                        self.record(episode)
                    if episode % 10_000 or episode + 1 == self.episodes:
                        self.SaveModels(episode)
                    break

    
    
    def Evaluate(self):
        total_reward = []
        episode_reward = 0
        succes_rate = []
        for episode in range(10):
            step = self.env.reset()
            step = self.norm.normalize_state(step)
            episode_reward = 0
            for t in range(self.env_params['max_timesteps']): 
                action = self.Action(step,param_noise=None, batch=True)
                nextstate, reward, done, info = self.env.step(action)
                nextstate = self.norm.normalize_state(nextstate)
                reward = self.norm.normalize_reward(reward)
                episode_reward += reward[:]
                if t + 1 == self.env_params['max_timesteps']:
                    succes_rate.append([i['is_success'] for i in info])
                    total_reward.append(episode_reward)
                    episode_reward = 0
        total_reward = np.array(total_reward).flatten()
        succes_rate = np.array(succes_rate).flatten()
        average_reward = sum(total_reward)/len(total_reward)
        min_reward = min(total_reward)
        max_reward = max(total_reward)
        succes = np.sum(succes_rate)/len(succes_rate)
        self.tensorboard.update_stats(Succes_Rate=succes,reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward)

    
    
    def record(self, episode):
        try:
            if not os.path.exists("videos"):
                os.mkdir('videos')
            recorder = VideoRecorder(self.evaluate_env, path=f'videos/episode-{episode}.mp4')
            for _ in range(5):
                done =False
                step = self.evaluate_env.reset()
                step = self.norm.normalize_state(step)
                while not done:
                    recorder.capture_frame()
                    action = self.Action(step, param_noise=None, batch=False)
                    nextstate,reward,done,info = self.evaluate_env.step(action)
                    nextstate = self.norm.normalize_state(nextstate)
                    reward = self.norm.normalize_reward(reward)
                    state = nextstate
            recorder.close()
        except Exception as e:
            print(e)

    
    def SaveModels(self, ep):
        torch.save(self.actor.state_dict(), self.path+ f"/Actor-{ep}.pt")
        torch.save(self.critic.state_dict(), self.path+ f"/Critic-{ep}.pt")

    
    def LoadModels(self, actorpath, criticpath):
        actor = Actor(self.env_params, self.hidden_neurons)
        critic  = Critic(self.env_params, self.hidden_neurons)
        actor.load_state_dict(torch.load(actorpath))
        critic.load_state_dict(torch.load(criticpath))
        return actor, critic

    
    def SoftUpdateTarget(self, network, target):
        for param, target_param in zip(network.parameters(), target.parameters()):
             target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

def get_params(env):
	obs = env.reset()
	params = {'observation': obs['observation'].shape[0],
			'goal': obs['desired_goal'].shape[0],
			'action': env.action_space.shape[0],
			'max_action': env.action_space.high[0],
			}
	params['max_timesteps'] = env._max_episode_steps
	return params

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', type=int, default=20000, help='number of episodes')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batch to pass through the network')
    parser.add_argument('--render', type=str2bool, default=False, help='whether or not to render the screen')
    parser.add_argument('--her', type=str2bool, default=True, help='Hindsight experience replay')
    parser.add_argument('--per', type=str2bool, default=True, help='Prioritized experience replay')
    parser.add_argument('--tb', type=str2bool, default=False, help='tensorboard activated via code')
    args = parser.parse_args()

    num_threads = os.cpu_count() -2
    iteration = 1
    env = gym.make('FetchReach-v1') 
    env_make = tuple(lambda: gym.make('FetchReach-v1') for _ in range(num_threads))
    envs = SubprocVecEnv(env_make)
    env_param = get_params(env)
    agent = Agent(env, envs,env_param,n_episodes=args.n_episodes, n_threads=num_threads, save_path=None, \
    batch_size=args.batch_size, tensorboard=args.tb ,her=args.her, per=args.per ,screen=args.render)
    agent.Explore(iteration)

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import argparse
from copy import deepcopy
import os
import gym
from models import Actor
import torch
import matplotlib.pyplot as plt


def Normalize(state, o_mean, o_std, g_mean, g_std):
    o = state['observation']
    g = state['desired_goal']
    o_clip = np.clip(o, -CLIP_OBS, CLIP_OBS)
    g_clip = np.clip(g, -CLIP_OBS, CLIP_OBS)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -CLIP_RANGE, CLIP_RANGE)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -CLIP_RANGE, CLIP_RANGE)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


def get_params(env):
	obs = env.reset()
	params = {'observation': obs['observation'].shape[0],
			'goal': obs['desired_goal'].shape[0],
			'action': env.action_space.shape[0],
			'max_action': env.action_space.high[0],
			}
	params['max_timesteps'] = env._max_episode_steps
	return params

CLIP_OBS = 200
CLIP_RANGE = 5

parser = argparse.ArgumentParser()
parser.add_argument('--numb-tries',type=int, default=20, help='number of episodes to run')
parser.add_argument('--numb-runs',type=int, default=10, help='number of episodes to run')
args = parser.parse_args()

normal = os.path.join("models/HandManipulateBlock-v0/normal/" "model.pt")
distance = os.path.join("models/HandManipulateBlock-v0/distance/" "model.pt")
impact = os.path.join("models/HandManipulateBlock-v0/impact/" "model.pt")

agents = [normal, distance, impact]

env = gym.make('HandManipulateBlock-v0')
env_params = get_params(env)


agent = Actor(env_params, 256)

overal_score = []
for path in agents:
    mean_obs, std_obs, mean_g, std_g, model = torch.load(path, map_location=lambda storage, loc: storage)
    agent.load_state_dict(model)
    success_rate = []
    for __ in range(args.numb_runs):
        success = []
        for __ in range(args.numb_tries):
            state = env.reset()
            state = Normalize(state, mean_obs, std_obs, mean_g, std_g)
            
            for _ in range (env._max_episode_steps):
                with torch.no_grad():
                    action = agent.forward(state)
                action = action.detach().numpy().squeeze()
                new_state, reward, _, info = env.step(action)
                new_state = Normalize(new_state, mean_obs, std_obs, mean_g, std_g)
                state = new_state
            else:
                if info['is_success']:
                    success.append(True)
        else:
            success_rate.append(np.sum(success)/args.numb_tries)
            success = []
    else: 
        overal_score.append(deepcopy(success_rate))
        success_rate = []



plt.boxplot(overal_score)
plt.xticks([1,2,3],["normal", "distance", "impact"])
plt.title("Succes Rate")
plt.figure(figsize = (4,4))
plt.savefig('data/boxplots.png')
plt.show()

print(f'normal={np.mean(overal_score[0])}, distance={np.mean(overal_score[1])}, impact={np.mean(overal_score[2])} \n\n')

groups = [['normal']*args.numb_runs, ['distance']*args.numb_runs, ['impact']*args.numb_runs]
F, p = stats.f_oneway(overal_score[0], overal_score[1], overal_score[2])
print(f'F-score = {F}, p={p} \n\n')

post_hoc = pairwise_tukeyhsd(np.concatenate(overal_score), np.concatenate(groups))
print(post_hoc)
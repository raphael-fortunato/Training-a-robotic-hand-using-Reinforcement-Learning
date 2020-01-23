import gym
import numpy as np
import torch
import argparse
import os 

from models import Actor

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo-length',type=int, default=200000, help='number of demo episodes to run')
    parser.add_argument('--distance', action='store_true', help='shows model with the distance version of per')
    parser.add_argument('--impact', action='store_true', help='shows model with the impact version of per')
    args = parser.parse_args()
    if args.distance:
        model_path = os.path("models/HandManipulateBlock-v0/distance/model.pt")
    elif args.impact:
        model_path = os.path("models/HandManipulateBlock-v0/impact/model.pt")
    else:
        model_path = os.path("models/HandManipulateBlock-v0/normal/model.pt")

    env = gym.make('HandManipulateBlock-v0')
    env_params = get_params(env)
    
    mean_obs, std_obs, mean_g, std_g, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    agent = Actor(env_params, 256)
    agent.load_state_dict(model)

    for __ in range(args.demo_length):
        state = env.reset()
        state = Normalize(state, mean_obs, std_obs, mean_g, std_g)
        for _ in range (env._max_episode_steps):
            env.render()
            with torch.no_grad():
                action = agent.forward(state)
            action = action.detach().numpy().squeeze()
            new_state, reward, _, info = env.step(action)
            new_state = Normalize(new_state, mean_obs, std_obs, mean_g, std_g)
            state = new_state


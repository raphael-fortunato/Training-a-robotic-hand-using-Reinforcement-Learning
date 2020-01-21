import gym
import numpy as np
import torch

from models import Actor

def Normalize(state, mean_obs, std_obs, mean_g, std_g):
    obs = state['observation']
    g = state['desired_goal']
    obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
    g = np.clip(g, -CLIP_OBS, CLIP_OBS)
    obs = np.clip((obs - mean_obs) / std_obs, -CLIP_RANGE, CLIP_RANGE)
    g = np.clip((g - mean_g)/ mean_g, -CLIP_RANGE, CLIP_RANGE)
    return torch.tensor(np.concatenate([obs, g]), dtype=torch.float)

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
    env = gym.make('HandManipulateBlock-v0')
    env_params = get_params(env)
    model_path = "saved_models/HandManipulateBlock-v0/model.pt"
    mean_obs, std_obs, mean_g, std_g, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    agent = Actor(env_params, 256)
    agent.load_state_dict(model)

    while True:
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


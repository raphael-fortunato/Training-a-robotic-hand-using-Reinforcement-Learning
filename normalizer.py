import numpy as np
from library.stable_baselines.common.running_mean_std import RunningMeanStd
import pdb

class Normalizer:
    """
    Normalizes state and vectors through running means and running stds. Based on open ai's stable baselines
    """
    def __init__(self, obs_shape, gamma, training=True ,clip_obs = 5, clip_rew=5, eps=1e-8):
        self.runn_mean_obs = RunningMeanStd(shape=obs_shape['observation'])
        self.runn_mean_des = RunningMeanStd(shape=obs_shape['goal'])
        self.runn_mean_ach = RunningMeanStd(shape=obs_shape['goal'])
        self.runn_mean_reward = RunningMeanStd(shape=(1,))
        self.clip_obs = clip_obs
        self.clip_rew =clip_rew
        self.epsilon  = eps
        self.training = training
        self.disc_reward =np.zeros((1,))
        self.gamma =gamma


    def normalize_state(self, obs):
        observation = obs['observation']
        desired_goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']
        
        if self.training:
            self.runn_mean_obs.update(obs['observation'])
            self.runn_mean_des.update(obs['desired_goal'])
            self.runn_mean_ach.update(obs['achieved_goal'])

        observation = np.clip((observation - self.runn_mean_obs.mean) / np.sqrt(self.runn_mean_obs.var +self.epsilon), -self.clip_obs, self.clip_obs)
        desired_goal = np.clip((desired_goal - self.runn_mean_des.mean) / np.sqrt(self.runn_mean_des.var +self.epsilon), -self.clip_obs, self.clip_obs)
        achieved_goal = np.clip((achieved_goal - self.runn_mean_ach.mean) / np.sqrt(self.runn_mean_ach.var +self.epsilon), -self.clip_obs, self.clip_obs)

        obs['observation'] = observation
        obs['desired_goal'] = desired_goal
        obs['achieved_goal'] = achieved_goal

        return obs


    def normalize_reward(self, reward):
        if self.training:
            self.disc_reward = self.disc_reward * self.gamma +reward
            self.runn_mean_reward.update(self.disc_reward)

        r = np.clip(reward / np.sqrt(self.runn_mean_reward.var + self.epsilon), -self.clip_rew, self.clip_rew)

        return r

    
    def load(load_path, venv):
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return: (VecNormalize)
        """
        with open(load_path, "rb") as file_handler:
            norm = pickle.load(file_handler)

        return norm

    def save(self, save_path):
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

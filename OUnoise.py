import numpy as np


class OUnoise:
    """
    Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

    def __init__(self, env_param, mu=0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=800):
        self.mu = mu
        self.theta =theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        
        self.action_dim = env_param['action']
        self.low = - env_param['max_action']
        self.high = env_param['max_action']
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def __call__(self):
        ou_state = self.evolve_state()
        self.sigma =  self.max_sigma - (self.max_sigma - self.min_sigma) 
        return ou_state

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    


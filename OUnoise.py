

class OUnoise:
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


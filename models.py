import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class Actor(nn.Module):
    def __init__(self,env_params, hidden_neurons):
       super(Actor, self).__init__() 
       self.max_action = env_params['max_action']
       self.fc1 = nn.Linear(env_params['observation']+ env_params['goal'],hidden_neurons )
       self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
       self.fc3 = nn.Linear(hidden_neurons, hidden_neurons)
       self.output = nn.Linear(hidden_neurons, env_params['action'])


    def forward(self, x):


        if torch.cuda.is_available():
            x = x.cuda()
        else:
            x = torch.tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = self.max_action * torch.tanh(self.output(x))
        return action


class Critic(nn.Module):
    def __init__(self ,env_params, hidden_neurons):
        super(Critic, self).__init__()
        self.max_action = env_params['max_action']
        self.fc1 = nn.Linear(env_params['observation'] + env_params['goal']+ env_params['action'],hidden_neurons )
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.q = nn.Linear(hidden_neurons, 1)

    def forward(self, x, action):
        x = torch.cat([x, action/ self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q(x)
        return q_value
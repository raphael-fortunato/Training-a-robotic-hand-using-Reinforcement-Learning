import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class Actor(nn.Module):
    def __init__(self,env_params, hidden_neurons):
        super(Actor, self).__init__() 
        self.max_action = env_params['max_action']
        self.fc1 = nn.Linear(env_params['observation']+ env_params['goal'],hidden_neurons )
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.action_out = nn.Linear(hidden_neurons, env_params['action'])
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1,f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2,f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc3.weight.data, -f3,f3)
        torch.nn.init.uniform_(self.fc3.bias.data, -f3, f3)

    def forward(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
           
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = self.max_action * torch.tanh(self.action_out(x))
        return action


class Critic(nn.Module):
    def __init__(self ,env_params, hidden_neurons):
        super(Critic, self).__init__()
        self.max_action = env_params['max_action']
        self.fc1 = nn.Linear(env_params['observation'] + env_params['goal']+ env_params['action'],hidden_neurons )
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.q_out = nn.Linear(hidden_neurons, 1)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1,f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2,f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc3.weight.data, -f3,f3)
        torch.nn.init.uniform_(self.fc3.bias.data, -f3, f3)        

    def forward(self, x, action):
        x = torch.cat([x, action/ self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
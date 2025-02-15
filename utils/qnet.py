import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Qnet(nn.Module):
    def __init__(self, state_size, action_size, hiden_dim, seed):
        super(Qnet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hiden_dim)
        self.fc2 = nn.Linear(hiden_dim, hiden_dim)
        self.fc3 = nn.Linear(hiden_dim, action_size)
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
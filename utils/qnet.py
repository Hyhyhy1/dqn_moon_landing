import torch
import torch.nn as nn

class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim, hiden_dim):
        super(Qnet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hiden_dim),
            nn.ReLU(),
            nn.Linear(hiden_dim, hiden_dim),
            nn.ReLU(),
            nn.Linear(hiden_dim, action_dim),
            nn.ReLU())

    def forward(self, state):
        return self.model(state)
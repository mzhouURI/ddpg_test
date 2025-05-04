import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, error_dim, action_dim):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + error_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=1)
        return self.q_net(x)

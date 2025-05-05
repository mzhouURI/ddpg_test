import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim = 64):
        super().__init__()
        self.net = nn.Sequential(
                                nn.Linear(state_dim, hidden_dim), 
                                nn.ReLU()
                                )
        self.pi = nn.Sequential(
                                nn.Linear(hidden_dim, hidden_dim), 
                                nn.ReLU(), 
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(), 
                                nn.Linear(hidden_dim, act_dim)
                                )
        self.v = nn.Sequential(
                                nn.Linear(hidden_dim, hidden_dim), 
                                nn.ReLU(), 
                                nn.Linear(hidden_dim, 1)
                                )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, state):
        x = self.net(state)
        return self.pi(x), self.v(x)

    def act(self, state):
        mean, _ = self.forward(state)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1)

    def evaluate(self, state, act):
        mean, value = self.forward(state)
        dist = Normal(mean, self.log_std.exp())
        logp = dist.log_prob(act).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value.squeeze()

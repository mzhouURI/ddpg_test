import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from model import ActorCritic
import numpy as np
import statistics

class PPOAgent:
    def __init__(self, state_dim, act_dim, hidden_dim =64, gamma=0.99, lam=0.95, clip=0.2, lr=3e-4, epochs=10):
        self.model = ActorCritic(state_dim, act_dim, hidden_dim)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Buffers for one episode or a fixed horizon
        self.reset_buffer()

    def reset_buffer(self):
        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.done_buf = []

    def select_action(self, state_np):
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)  # add batch dim
        with torch.no_grad():
            mean, _ = self.model.forward(state)
            std = self.model.log_std.exp()
            dist = torch.distributions.Normal(mean, std)

            raw_action = dist.rsample()  # pre-tanh action
            action = torch.tanh(raw_action)

            # Change of variable (log probability correction for tanh squashing)
            logp = dist.log_prob(raw_action).sum(axis=-1)
            logp -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(axis=-1)

            # Value function estimate
            value = self.model.v(self.model.net(state))

        return action.squeeze(0).numpy(), logp.item(), value.item()



    def observe(self, state, action, reward, logp, value, done):
        """Store one transition"""
        self.obs_buf.append(state)
        self.act_buf.append(action)
        self.rew_buf.append(reward)
        self.logp_buf.append(logp)
        self.val_buf.append(value)
        self.done_buf.append(done)

    def compute_gae(self, last_val=0):
        adv_buf = []
        gae = 0
        val_buf = self.val_buf + [last_val]
        for t in reversed(range(len(self.rew_buf))):
            delta = self.rew_buf[t] + self.gamma * val_buf[t + 1] * (1 - self.done_buf[t]) - val_buf[t]
            gae = delta + self.gamma * self.lam * (1 - self.done_buf[t]) * gae
            adv_buf.insert(0, gae)
        ret_buf = [a + v for a, v in zip(adv_buf, self.val_buf)]
        return torch.tensor(adv_buf, dtype=torch.float32), torch.tensor(ret_buf, dtype=torch.float32)

    def train(self):
        """Train PPO using the collected buffer."""
        obs_np = np.array(self.obs_buf)
        obs = torch.tensor(obs_np, dtype=torch.float32)

        act_np = np.array(self.act_buf)
        act = torch.tensor(act_np, dtype=torch.float32)
        logp_old = torch.tensor(self.logp_buf, dtype=torch.float32)

        adv, ret = self.compute_gae()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss_stat = []
        for _ in range(self.epochs):
            logp, entropy, value = self.model.evaluate(obs,act)
            ratio = torch.exp(logp - logp_old)

            surrogate1 = ratio * adv
            surrogate2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = F.mse_loss(value, ret)
            entropy_bonus = entropy.mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
            loss_stat.append(policy_loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.reset_buffer()  # clear for next batch
        return torch.mean(torch.stack(loss_stat))
    
    def save_model(self):
        """
        Save the actor and critic models to disk.
        """
        # Save the actor model
        torch.save(self.model.state_dict(), "model/PPO.pth")
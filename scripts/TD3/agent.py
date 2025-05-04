import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import deque
from critic import Critic
from siamese import SiamesePoseControlNet


class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)

    def add(self, s, g, a, r, s2, g2, d):
        self.buffer.append((s, g, a, r, s2, g2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, g, a, r, s2, g2, d = map(np.stack, zip(*batch))
        return map(torch.FloatTensor, (s, g, a, r, s2, g2, d))

class TD3Agent:
    def __init__(self, state_dim, error_state_dim, action_dim, max_action, device='cpu', actor_ckpt=None):
        # Initialize actor (SiamesePoseControlNet) and critics (Critic)
        self.actor = SiamesePoseControlNet(current_pose_dim = state_dim, goal_pose_dim =  error_state_dim, latent_dim = 64, thruster_num=action_dim)

        if actor_ckpt is not None:
            self.actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
            print(f"Loaded actor weights from {actor_ckpt}")

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        # Initialize critic networks and target critics
        self.critic1 = Critic(state_dim, error_state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, error_state_dim, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=1e-3)

        # Set max action for normalizing outputs
        self.max_action = max_action

        # Initialize replay buffer for storing experiences
        self.replay_buffer = ReplayBuffer()

        self.device = device

    def select_action(self, state, error_state, noise_std=0.1):
        """
        Select action based on the current state and error_state (difference between current and goal).
        Add noise for exploration during training.
        """
        # state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state = state.detach().clone().float().to(self.device)
        error_state = error_state.detach().clone().float().to(self.device)

        # Get action from the actor (SiamesePoseControlNet)
        action = self.actor(state, error_state)

        # Add noise for exploration
        action = action + noise_std * torch.randn_like(action)
        return action.clamp(-self.max_action, self.max_action)

    def train(self, batch_size=64, gamma=0.99, tau=0.005):
        """
        Train the TD3 agent (actor and critics) using a batch of experiences.
        """
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(batch_size)
        state, error_state, action, reward, next_state, next_error_state, done = batch

        state = state.detach().clone().float().to(self.device)
        error_state = error_state.detach().clone().float().to(self.device)
        action = action.detach().clone().float().to(self.device)
        reward = reward.detach().clone().float().to(self.device)
        next_state = next_state.detach().clone().float().to(self.device)
        next_error_state = next_error_state.detach().clone().float().to(self.device)
        done = done.detach().clone().float().to(self.device)

        # Compute target Q-values using the target critics
        with torch.no_grad():
            # next_action = self.actor_target(next_state, next_error_state)
            # Ensure both tensors have the correct shape
            next_state = next_state.view(-1, next_state.size(-1))  # Flatten the batch dimension, keeping the feature dimension
            next_error_state = next_error_state.view(-1, next_error_state.size(-1))  # Similarly for the error_state

            # Then pass them to the actor target
            next_action = self.actor_target(next_state, next_error_state)

            target_q1 = self.critic1_target(next_state, next_error_state, next_action)
            target_q2 = self.critic2_target(next_state, next_error_state, next_action)
            target_q = reward + (1 - done) * gamma * torch.min(target_q1, target_q2)

        # Update the critics
        
        state = state.view(state.size(0), -1)  # Flatten all dimensions except the batch size
        error_state = error_state.view(error_state.size(0), -1)
        action = action.view(action.size(0), -1)


        q1 = self.critic1(state, error_state, action)
        q2 = self.critic2(state, error_state, action)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        # Optimize the critics
        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic_optimizer.step()

        # Update the actor every few steps
        if torch.randint(0, 2, (1,)).item() == 0:
            # Get the action from the actor
            actor_loss = -self.critic1(state, error_state, self.actor(state, error_state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

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
    def __init__(self, state_dim, error_state_dim, action_dim, max_action, device='cpu', actor_ckpt=None, actor_lr = 1e-4, critic_lr = 1e-4):
        # Initialize actor (SiamesePoseControlNet) and critics (Critic)
        self.actor = SiamesePoseControlNet(current_pose_dim = state_dim, goal_pose_dim =  error_state_dim, latent_dim = 64, thruster_num=action_dim)
        
        if actor_ckpt is not None:
            self.actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
            print(f"Loaded actor weights from {actor_ckpt}")

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, amsgrad = True, weight_decay=1e-4)

        # Initialize critic networks and target critics
        self.critic1 = Critic(state_dim, error_state_dim, action_dim).to(device)
        self.critic1.apply(self.critic1.init_weights)
        self.critic2 = Critic(state_dim, error_state_dim, action_dim).to(device)
        self.critic2.apply(self.critic2.init_weights)

        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr,  amsgrad = True, weight_decay=1e-4)

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
            next_state = next_state.squeeze(1)  # Now shape is [64, 10]
            next_error_state = next_error_state.squeeze(1)  # Now shape is [64, 4]
            next_action = self.actor_target(next_state, next_error_state)
            target_q1 = self.critic1_target(next_state, next_error_state, next_action)
            target_q2 = self.critic2_target(next_state, next_error_state, next_action)
            target_q1 = target_q1.squeeze(1)  # [64]
            target_q2 = target_q2.squeeze(1)  # [64]
            target_q = reward + (1 - done) * gamma * torch.min(target_q1, target_q2)
            target_q = target_q.unsqueeze(1)
            # target_q = target_q.clamp(-10, 10)
        # Update the critics
        
        state = state.squeeze(1)
        error_state = error_state.squeeze(1)
        action = action.squeeze(1)

        q1 = self.critic1(state, error_state, action)
        q2 = self.critic2(state, error_state, action)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        print(f"Average Q1: {q1.mean().item():.4f}",
              f"Average Q2: {q2.mean().item():.4f}",
              f"Average TQ: {target_q.mean().item():.4f}",
              f"reward: {reward.mean().item():.4f}")


        # Optimize the critics
        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic_optimizer.step()

        # Update the actor every few steps
        actor_loss = 0
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

        return critic1_loss, critic2_loss, actor_loss
    

    
    def save_model(self, save_dir="", actor_filename="actor.pth", critic1_filename="critic1.pth", critic2_filename="critic2.pth"):
        """
        Save the actor and critic models to disk.
        """
        # Save the actor model
        torch.save(self.actor.state_dict(), "model/actor.pth")

        # Save the critic models
        torch.save(self.critic1.state_dict(), "model/critic1.pth")
        torch.save(self.critic2.state_dict(), "model/critic2.pth")

        # Optionally, save the target models as well
        torch.save(self.actor_target.state_dict(), "model/actor_target.pth")
        torch.save(self.critic1_target.state_dict(), "model/critic1_target.pth")
        torch.save(self.critic2_target.state_dict(), "model/critic2_target.pth")

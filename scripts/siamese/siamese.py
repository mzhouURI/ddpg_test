import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PoseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(PoseEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, pose):
        x = F.relu(self.fc1(pose))
        # x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.manual_seed(42)  # Fix seed per layer for repeatability
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class SiamesePoseControlNet(nn.Module):
    def __init__(self, current_pose_dim=3, goal_pose_dim=2, latent_dim=32, thruster_num =4):
        super(SiamesePoseControlNet, self).__init__()
        self.current_encoder = PoseEncoder(input_dim=current_pose_dim, latent_dim=latent_dim)
        self.current_encoder.apply(self.current_encoder.init_weights)

        self.goal_encoder = PoseEncoder(input_dim=goal_pose_dim, latent_dim=latent_dim)
        self.current_encoder.apply(self.current_encoder.init_weights)

        self.control_head = nn.Sequential(
            nn.Linear(2 * latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, thruster_num)  # Output: [v_linear, v_angular, etc.]
            # nn.Tanh(),
        )

    def forward(self, current_pose, goal_pose):
        current_feat = self.current_encoder(current_pose)
        goal_feat = self.goal_encoder(goal_pose)
        combined = torch.cat((current_feat, goal_feat), dim=1)
        control = self.control_head(combined)
        return control
    
    
class OnlineTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.loss_fn = nn.MSELoss()

    # def train(self, predicted_control, true_control):
    #     if isinstance(predicted_control, list):
    #         predicted_control = torch.tensor(predicted_control, dtype=torch.float32)
    #     if isinstance(true_control, list):
    #         true_control = torch.tensor(true_control, dtype=torch.float32)

    #     # Compute loss (MSE between predicted control and true control)
    #     loss = self.loss_fn(predicted_control, true_control)

    #     # Backward pass and optimization
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     return loss.item()
    def train(self, predicted_control, true_control, error_pose):
        # Ensure both inputs are torch tensors and of correct type
        if isinstance(predicted_control, list):
            predicted_control = torch.tensor(predicted_control, dtype=torch.float32)
        if isinstance(true_control, list):
            true_control = torch.tensor(true_control, dtype=torch.float32)

        # Ensure inputs are 2D (batch_size, control_dim), if they are not already
        if predicted_control.ndimension() == 1:  # If it's 1D (e.g., single sample)
            predicted_control = predicted_control.unsqueeze(0)

        if true_control.ndimension() == 1:  # If it's 1D (e.g., single sample)
            true_control = true_control.unsqueeze(0)

        if error_pose.ndimension() == 1:
            error_pose = error_pose.unsqueeze(0)

        # Make sure the dimensions of predicted and true control match
        assert predicted_control.shape == true_control.shape, \
            f"Shape mismatch: predicted {predicted_control.shape}, true {true_control.shape}"
        error_pose = np.array(error_pose)
        # Compute MSE loss between the batch of predicted and true control commands
        min_val = np.min(error_pose, axis=0)  # shape (4,)
        max_val = np.max(error_pose, axis=0)  # shape (4,)
        scaled_error = (error_pose - min_val) / (max_val - min_val)
        overall_mean = np.mean(scaled_error)               # total mean
        loss = self.loss_fn(predicted_control, true_control) + 0.1*overall_mean

        # Perform the backward pass and optimize
        self.optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Backpropagate gradients
        self.optimizer.step()  # Perform an optimization step

        return loss.item()
    
    def save_model(self):
        torch.save(self.model.state_dict(), "siamese_pose_control_net.pth")
        
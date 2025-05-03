import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def train(self, predicted_control, true_control):
        if isinstance(predicted_control, list):
            predicted_control = torch.tensor(predicted_control, dtype=torch.float32)
        if isinstance(true_control, list):
            true_control = torch.tensor(true_control, dtype=torch.float32)

        # Compute loss (MSE between predicted control and true control)
        loss = self.loss_fn(predicted_control, true_control)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def save_model(self):
        torch.save(self.model.state_dict(), "siamese_pose_control_net.pth")
        
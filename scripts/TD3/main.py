import time
import numpy as np
from agent import TD3Agent
import rclpy
from rclpy.node import Node
from mvp_msgs.msg import ControlProcess
from std_srvs.srv import SetBool
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64, Float64MultiArray
import torch
from collections import deque

class TD3_ROS(Node):
    def __init__(self):
        super().__init__('ddpg_node')
        self.set_point_interval = 100
        #initial set point
        
        self.subscription = self.create_subscription(ControlProcess, '/mvp2_test_robot/controller/process/value', self.state_callback, 1)
        self.subscription2 = self.create_subscription(ControlProcess, '/mvp2_test_robot/controller/process/error', self.state_error_callback, 1)


        self.set_point_pub = self.create_publisher(ControlProcess, '/mvp2_test_robot/controller/process/set_point', 3)
        
        self.loss_pub = self.create_publisher(Float64MultiArray, '/training/loss',10)
        self.total_reward_pub = self.create_publisher(Float64, '/training/episode_reward', 10)
                 #initial set point
        self.set_point = ControlProcess()
        self.set_point.orientation.x = 0.0
        self.set_point.orientation.y = 0.0
        self.set_point.velocity.y = 0.0
        self.set_point.header.frame_id = "mvp2_test_robot/world"
        self.set_point.child_frame_id = "mvp2_test_robot/base_link"
        self.set_point.control_mode = "4dof"
        self.set_point.position.z = 0.0
        self.set_point.orientation.z = 0.0
        self.set_point.velocity.x = 0.0
        self.set_counter = 1  # Start from 0

        self.thrust_cmd = [0,0]
        ##setting mode
        self.training = True
        self.training_episode = 0
        self.buffer = deque(maxlen=10000)  # or any reasonable size
        self.batch_size =64

        state = {
            'z': (0),
            'euler': (0,0,0), # Quaternion (x, y, z, w)
            'uvw': (0,0,0),
            'pqr': (0,0,0)
        }
        error_state = {
             'z': (0),
            'euler': (0,0), # Quaternion (x, y, z, w)
            'u': (0)
        }
        self.state = self.flatten_state(state)
        self.error_state = self.flatten_state(error_state)

        device = torch.device("cpu")
        self.model = TD3Agent(len(self.state), len(self.error_state), 4, 1, device=device, 
                              actor_ckpt='05-03-siamese.pth',
                              actor_lr = 1e-6, critic_lr= 1e-7)
        self.total_reward = 0

        # self.timer_model_save = self.create_timer(100, self.save_model)
            
        self.thruster_pub = self.create_publisher(Float64MultiArray, '/mvp2_test_robot/stonefish/thruster_command', 5)

        self.timer_setpoint_update = self.create_timer(60, self.set_point_update)
        self.timer_setpoint_pub = self.create_timer(1.0, self.set_point_publish)
        self.timer_pub = self.create_timer(0.5, self.step)

        self.set_controller = self.create_client(SetBool, '/mvp2_test_robot/controller/set')  
        self.active_controller(True)
        
    def active_controller(self, req: bool):
        set_controller = SetBool.Request()
        set_controller.data = req
        future = self.set_controller.call_async(set_controller)
        # rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def set_point_update(self):
        print(f"episode reward = {self.total_reward}")
        self.model.save_model()
        msg = Float64()
        msg.data = float (self.total_reward)
        self.total_reward_pub.publish(msg)
        #update setpoint
        self.set_point.position.z = random.uniform(-5,-1)
        self.set_point.orientation.z = random.uniform(-3.14, 3.14)
        self.set_point.velocity.x = random.uniform(0.0, 0.3)
        self.total_reward = 0
    
    def set_point_publish(self):
        self.set_point_pub.publish(self.set_point)

    def wrap_to_pi(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def flatten_state(self, state_dict):
        def flatten_value(v):
            if isinstance(v, (tuple, list, set, np.ndarray)):
                result = []
                for item in v:
                    result.extend(flatten_value(item))  # recursive call
                return result
            else:
                return [v]

        flat = []
        for value in state_dict.values():
            flat.extend(flatten_value(value))
        
        return np.array(flat, dtype=np.float32)

    def state_callback(self, msg):

        state = {
            'z': (msg.position.z),
            'euler': (msg.orientation.x, msg.orientation.y, msg.orientation.z), 
            'uvw': (msg.velocity.x, msg.velocity.y, msg.velocity.z),
            'pqr': {msg.angular_rate.x, msg.angular_rate.y, msg.angular_rate.z}
        }
       
        self.state = self.flatten_state(state)

    def state_error_callback(self, msg):

        state_error = {
            'z': (msg.position.z),
            'euler': (msg.orientation.y, msg.orientation.z),  
            'u': (msg.velocity.x)
        }
        self.error_state = self.flatten_state(state_error)

    # def save_model(self):

    def step(self):
        # try:
            current_pose = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
            error_pose = torch.tensor(self.error_state, dtype=torch.float32).unsqueeze(0)
            action = self.model.select_action(current_pose, error_pose, noise_std= 0.1)
            msg = Float64MultiArray()
            msg.data = action.detach().cpu().numpy().flatten().tolist()                   
            self.thruster_pub.publish(msg)
            time.sleep(0.2)

            new_pose = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
            new_error_pose = torch.tensor(self.error_state, dtype=torch.float32).unsqueeze(0)
            
            #error state reward
            error_magnitude = torch.norm(new_error_pose)

            # Reward is negative distance (minimizing the error is encouraged)
            reward = -error_magnitude.item()  # Convert tensor to scalar value
            
            done = False
            self.model.replay_buffer.add(current_pose, error_pose, action.detach().cpu().numpy(), reward, new_pose, new_error_pose, done)
            if len(self.model.replay_buffer.buffer) > self.batch_size:  # Start training after enough experiences
                c1_loss, c2_loss, actor_loss = self.model.train(batch_size=self.batch_size)
                msg = Float64MultiArray()
                msg.data = [float(c1_loss), float(c2_loss), float(actor_loss)]
                self.loss_pub.publish(msg)
            self.total_reward  = self.total_reward + reward
        # except Exception as e:
        #     self.get_logger().error(f"Error in step(): {e}")

def main(args=None):

    rclpy.init(args=args)
    node = TD3_ROS()
    node.set_point_update()

    try:
        rclpy.spin(node)  # This should keep the node alive and run timers and callbacks
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()

import time
import numpy as np
from ddpg import Agent
from utils import plot_learning_curve
import rclpy
from rclpy.node import Node
from mvp_msgs.msg import ControlProcess
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64

class DDPG_ROS(Node):
    def __init__(self):
        super().__init__('ddpg_node')
        self.set_point_interval = 100
        #initial set point
        self.set_point = Odometry()
        self.set_point.pose.pose.position.z = -2.0
        self.set_point.header.frame_id = "mvp2_test_robot/odom"
        self.set_point.child_frame_id = "mvp2_test_robot/base_link"
        self.set_point.pose.pose.orientation.x = 0
        self.set_point.pose.pose.orientation.y = 0
        self.set_point.pose.pose.orientation.z = 0
        self.set_point.pose.pose.orientation.w = 1
        self.desired_pitch = 0
        self.subscription = self.create_subscription(Odometry, '/mvp2_test_robot/odometry/filtered', self.state_callback, 1)

        self.heave_bow_pub = self.create_publisher(Float64, '/mvp2_test_robot/control/thruster/heave_bow', 1)
        self.heave_stern_pub = self.create_publisher(Float64, '/mvp2_test_robot/control/thruster/heave_stern', 1)


        # self.set_controller = self.create_client(SetBool, '/mvp2_test_robot/controller/set')  
        # self.active_controller(True)

        # self.timer_set_update = self.create_timer(self.set_point_interval, self.set_point_update)

        # self.ddpg_state = {
        #     'xyz': (0,0,0),
        #     'euler': (0,0,0),  # Quaternion (x, y, z, w)
        #     'uvw': (0,0,0),
        #     'pqr': (0, 0, 0),
        #     'e_xyz': (0, 0, 0),
        #     'e_euler': {0,0,0},  # Quaternion (x, y, z, w)
        #     'e_uvw': (0, 0, 0),
        #     'e_pqr': (0, 0, 0),
        # }

        ddpg_state = {
            'z': (0),
            'euler': (0, 0, 0),  # Quaternion (x, y, z, w)
            'uvw': (0, 0, 0),
            'pqr': (0, 0, 0),
            'e_z': (0),
            'e_pitch': {0},  # Quaternion (x, y, z, w)
        }
        self.ddpg_state = self.flatten_state(ddpg_state)
        self.agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=self.ddpg_state.shape, tau=0.001,
                    batch_size=1, fc1_dims=50, fc2_dims=200, fc3_dims = 50,
                    n_actions = 2)
        
    def set_point_update(self):
        self.set_point.pose.pose.position.z = random.uniform(-5,-2)
        print(self.set_point.pose.pose.position.z)
    
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
        # Position difference
        dx = self.set_point.pose.pose.position.x - msg.pose.pose.position.x
        dy = self.set_point.pose.pose.position.y - msg.pose.pose.position.y
        dz = self.set_point.pose.pose.position.z - msg.pose.pose.position.z

        # Orientation difference (quaternion)
        q1 = [
            self.set_point.pose.pose.orientation.x,
            self.set_point.pose.pose.orientation.y,
            self.set_point.pose.pose.orientation.z,
            self.set_point.pose.pose.orientation.w,
        ]
        q2 = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]

        euler1 = R.from_quat(q1).as_euler('xyz', degrees=False)
        euler2 = R.from_quat(q2).as_euler('xyz', degrees=False)
        self.desired_pitch = euler1[1]
        euler_error = [self.wrap_to_pi(e2 - e1) for e1, e2 in zip(euler1, euler2)]
        # Linear velocity difference
        du = self.set_point.twist.twist.linear.x - msg.twist.twist.linear.x
        dv = self.set_point.twist.twist.linear.y - msg.twist.twist.linear.y
        dw = self.set_point.twist.twist.linear.z - msg.twist.twist.linear.z

        # Angular velocity difference
        dp = self.set_point.twist.twist.angular.x - msg.twist.twist.angular.x
        dq = self.set_point.twist.twist.angular.y - msg.twist.twist.angular.y
        dr = self.set_point.twist.twist.angular.z - msg.twist.twist.angular.z

        ddpg_state = {
            'z': (msg.pose.pose.position.z),
            'euler': euler2,  # Quaternion (x, y, z, w)
            'uvw': (msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z),
            'pqr': (dp, dq, dr),
            'e_z': (dz),
            'e_pitch': (euler_error[1]),  # Quaternion (x, y, z, w)
        }

        self.ddpg_state = self.flatten_state(ddpg_state)

    def step(self, actions):
        observation = self.ddpg_state
        msg = Float64()
        msg.data = float(actions[0])
        self.heave_bow_pub.publish(msg)
        msg.data = float(actions[1])
        self.heave_stern_pub.publish(msg)

        time.sleep(0.5)
        observation_ = self.ddpg_state

        #calculate reward
        z_e = observation_[10]
        pitch_e = observation_[11]
        w = observation_[6]
        thruster_reward =  np.sum(np.square(actions))
        # print(f"thruster reward = {thruster_reward}")
        print(f"actions: {actions}")
        # print(z_e, pitch_e, w, thruster_reward)
        # depth_reward = 1 / (1 + np.exp(-abs(z_e))) -0.5
        # pitch_reward = 1 / (1 + np.exp(-abs(pitch_e))) -0.5
        # heave_reward = 1 / (1 + np.exp(-abs(w))) - 0.5
        # thruster_reward = 1 / (1 + np.exp(-thruster_reward)) -0.5

        # print(depth_reward, pitch_reward, heave_reward, thruster_reward)
        reward = -100*abs(z_e) - 50*abs(pitch_e) - 10*abs(w) 
        
        done = False

        if(reward >-0.1):
            done = True

        return observation_, reward, done


def main(args=None):
    rclpy.init(args=args)
    node = DDPG_ROS()
    n_games = 100
    filename = 'test' + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'
    score_history = []
    score = 0
    try:
        
        for i in range(n_games):
            observation = node.ddpg_state
            done = False
            node.agent.noise.reset()
            node.set_point_update() #update set point

            while not done:
                # print('in while')
                # rclpy.spin(node)
                rclpy.spin_once(node, timeout_sec=0.1)
                observation = node.ddpg_state
                action = node.agent.choose_action(observation)
                observation_, reward, done = node.step(action)
                node.agent.remember(observation, action, reward, observation_, done)
                node.agent.learn()
                score += reward
                print(f"Reward, {reward:.5f}, depth {observation[0]: .3f}, desired depth {node.set_point.pose.pose.position.z: .3f}", 
                      f"pitch: {observation[2]: .3f}, desired pitch: {node.desired_pitch: .3f}",
                      f"heave: {observation[6]: .3f}")

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
            
        node.agent.save_models()

        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

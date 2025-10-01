#!/usr/bin/env python3
"""
Gymnasium Environment for Drone Ball Catching
Compatible with Stable-Baselines3 for PPO training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import random


class DroneCatchEnv(gym.Env):
    """
    Custom Gymnasium environment for training a drone to catch falling balls.

    Observation Space:
        - Drone position (x, y, z): 3 values
        - Drone velocity (vx, vy, vz): 3 values
        - Drone orientation (quaternion): 4 values
        - Ball position (x, y, z): 3 values
        - Ball velocity (vx, vy, vz): 3 values
        Total: 16 dimensions

    Action Space:
        - Continuous actions: [thrust, roll, pitch, yaw]
        - Each in range [-1, 1]

    Reward:
        - +10 for catching the ball
        - -1 for missing the ball
        - -0.01 * distance_to_ball (encourages pursuit)
        - -0.001 * action_magnitude (encourages smooth control)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, model_path="drone_model.xml", render_mode=None, max_steps=1000):
        super(DroneCatchEnv, self).__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Rendering
        self.render_mode = render_mode
        self.renderer = None
        self.viewer = None

        # Control parameters
        self.thrust_base = 3.2
        self.control_gains = {"roll": 2.0, "pitch": 2.0, "yaw": 1.5, "thrust": 3.0}

        # Episode parameters
        self.max_steps = max_steps
        self.current_step = 0
        self.ball_active = False
        self.episode_reward = 0.0

        # Define action and observation spaces
        # Actions: [thrust, roll, pitch, yaw] all in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Observations: drone state (10) + ball state (6) = 16
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32
        )

        # Get body IDs
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")

        print("✅ Drone Catch Environment initialized")

    def _get_obs(self):
        """Get current observation."""
        # Drone state
        drone_pos = self.data.qpos[0:3].copy()
        drone_quat = self.data.qpos[3:7].copy()
        drone_vel = self.data.qvel[0:3].copy()

        # Ball state
        ball_qpos_idx = 7
        ball_pos = self.data.qpos[ball_qpos_idx:ball_qpos_idx+3].copy()
        ball_vel = self.data.qvel[7:10].copy()

        # Construct observation
        obs = np.concatenate([
            drone_pos,      # 3
            drone_vel,      # 3
            drone_quat,     # 4
            ball_pos,       # 3
            ball_vel        # 3
        ]).astype(np.float32)

        return obs

    def _get_info(self):
        """Get additional info (for debugging)."""
        drone_pos = self.data.qpos[0:3]
        ball_qpos_idx = 7
        ball_pos = self.data.qpos[ball_qpos_idx:ball_qpos_idx+3]
        distance = np.linalg.norm(drone_pos - ball_pos)

        return {
            'drone_pos': drone_pos.copy(),
            'ball_pos': ball_pos.copy(),
            'distance': distance,
            'ball_active': self.ball_active,
            'episode_reward': self.episode_reward
        }

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reset drone to starting position
        self.data.qpos[0:3] = [0.0, 0.0, 2.0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:7] = 0.0

        # Spawn ball at random position
        ball_qpos_idx = 7
        x = random.uniform(-2.0, 2.0)
        y = random.uniform(-2.0, 2.0)
        z = random.uniform(4.0, 6.0)

        self.data.qpos[ball_qpos_idx:ball_qpos_idx+3] = [x, y, z]
        self.data.qpos[ball_qpos_idx+3:ball_qpos_idx+7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[7:10] = [0.0, 0.0, -0.5]  # Slow downward velocity
        self.data.qvel[10:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # Reset episode state
        self.current_step = 0
        self.ball_active = True
        self.episode_reward = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one step in the environment."""
        # action = [thrust, roll, pitch, yaw], all in [-1, 1]
        # Scale down actions to prevent aggressive movements
        action_scale = 0.5
        target_thrust = float(action[0]) * action_scale
        target_roll = float(action[1]) * action_scale
        target_pitch = float(action[2]) * action_scale
        target_yaw = float(action[3]) * action_scale

        # Apply damping for stability
        angular_damping = 3.0
        linear_damping = 1.5
        vertical_damping = 2.0

        angular_vel = self.data.qvel[3:6]
        linear_vel = self.data.qvel[0:3]

        # Add damping to control targets
        target_pitch -= angular_vel[1] * angular_damping * 0.1
        target_pitch += linear_vel[0] * linear_damping * 0.1
        target_roll -= angular_vel[0] * angular_damping * 0.1
        target_roll += linear_vel[1] * linear_damping * 0.1
        target_yaw -= angular_vel[2] * angular_damping * 0.1
        target_thrust -= linear_vel[2] * vertical_damping * 0.1

        # Calculate motor outputs
        thrust = self.thrust_base + target_thrust * self.control_gains["thrust"]
        roll = target_roll * self.control_gains["roll"]
        pitch = target_pitch * self.control_gains["pitch"]
        yaw = target_yaw * self.control_gains["yaw"]

        # Quadrotor motor mixing
        motor1 = thrust + roll + pitch + yaw
        motor2 = thrust - roll + pitch - yaw
        motor3 = thrust - roll - pitch + yaw
        motor4 = thrust + roll - pitch - yaw

        motors = [max(0, min(m, 10.0)) for m in [motor1, motor2, motor3, motor4]]
        self.data.ctrl[:] = motors

        # Apply ball drag for slow falling
        ball_vel_idx = 7
        drag_coefficient = 0.95
        self.data.qvel[ball_vel_idx:ball_vel_idx+3] *= drag_coefficient

        # Step physics simulation
        mujoco.mj_step(self.model, self.data)

        # Calculate reward
        reward = 0.0
        terminated = False
        truncated = False

        # Get positions
        drone_pos = self.data.qpos[0:3]
        ball_qpos_idx = 7
        ball_pos = self.data.qpos[ball_qpos_idx:ball_qpos_idx+3]

        # REWARD SHAPING - Fixed to prevent crash exploitation

        # Small survival bonus (encourages staying alive)
        reward += 0.01

        # Distance-based reward (much smaller penalty)
        distance = np.linalg.norm(drone_pos - ball_pos)
        reward -= 0.001 * distance  # Reduced from 0.01

        # Height maintenance reward (encourages staying at reasonable altitude)
        target_height = 2.0
        height_error = abs(drone_pos[2] - target_height)
        reward -= 0.001 * height_error

        # Check for FAILURES FIRST (most negative rewards)
        # Check if drone crashed
        if drone_pos[2] < 0.3:
            reward = -100.0  # HUGE penalty - make crashing very bad
            terminated = True
            print("💥 CRASH!")

        # Check if drone flew too far away
        elif abs(drone_pos[0]) > 5.0 or abs(drone_pos[1]) > 5.0 or drone_pos[2] > 10.0:
            reward = -50.0  # Big penalty
            terminated = True
            print("🚀 Out of bounds!")

        # Check for catch or miss (only if not crashed)
        elif self.ball_active:
            relative_pos = ball_pos - drone_pos
            horizontal_dist = np.sqrt(relative_pos[0]**2 + relative_pos[1]**2)
            vertical_dist = relative_pos[2]

            # Check if caught
            if horizontal_dist < 0.15 and -0.25 < vertical_dist < -0.05:
                reward += 100.0  # Increased from 10
                terminated = True
                print(f"✅ CATCH! Distance: {distance:.2f}m")

            # Check if missed (ball hit ground)
            elif ball_pos[2] < 0.2:
                reward -= 10.0  # Increased penalty from 1
                terminated = True
                print(f"❌ MISS! Final distance: {distance:.2f}m")

        # Check step limit
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        self.episode_reward += reward

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)

            self.renderer.update_scene(self.data, camera="external_camera")
            return self.renderer.render()

        elif self.render_mode == "human":
            # Human rendering is handled by mujoco.viewer in the training script
            pass

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer = None


# Test the environment
if __name__ == "__main__":
    print("Testing DroneCatchEnv...")

    env = DroneCatchEnv()

    # Test reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    print(f"Info: {info}")

    # Test random actions
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")

        if terminated or truncated:
            obs, info = env.reset()
            print("Episode finished, resetting...")

    env.close()
    print("✅ Environment test complete!")

#!/usr/bin/env python3
"""
Drone Ball Catching Game - For RL Training
The drone must catch falling balls using its bottom catcher zone.
Rewards: +10 for catching, -1 for missing
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import random


class DroneCatchGame:
    def __init__(self, model_path: str = "drone_model.xml"):
        """Initialize drone catching game."""
        print("🎮 Loading Drone Catch Game...")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Control parameters
        self.thrust_base = 3.2
        self.control_gains = {"roll": 2.0, "pitch": 2.0, "yaw": 1.5, "thrust": 3.0}

        # Control state
        self.target_thrust = 0.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.keys_pressed = set()

        # Game state
        self.score = 0
        self.catches = 0
        self.misses = 0
        self.total_reward = 0.0

        # Ball spawning
        self.ball_spawn_interval = 20.0  # seconds
        self.last_spawn_time = 0
        self.ball_active = False
        self.ball_slow_fall = True  # Make ball fall slowly

        # Get body IDs
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")

        # Initialize
        self.reset_simulation()
        print("✅ Catch game ready!")
        print("🎯 Objective: Catch falling balls with the bottom of your drone!")

    def reset_simulation(self):
        """Reset drone and ball positions."""
        # Reset drone
        self.data.qpos[0:3] = [0.0, 0.0, 2.0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:7] = 0.0

        # Reset ball - put it on ground far away initially
        ball_qpos_idx = 7  # Ball starts after drone's 7 DOF
        self.data.qpos[ball_qpos_idx:ball_qpos_idx+3] = [5.0, 5.0, 0.1]  # On ground, far away
        self.data.qpos[ball_qpos_idx+3:ball_qpos_idx+7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[7:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.ball_active = False

    def spawn_ball(self):
        """Spawn a ball at a random position ON THE GROUND."""
        ball_qpos_idx = 7

        # Random spawn position on ground (within 3 meter radius)
        x = random.uniform(-3.0, 3.0)
        y = random.uniform(-3.0, 3.0)
        # Floor surface is at z=0, ball radius=0.1, so center at z=0.1
        z = 0.1

        self.data.qpos[ball_qpos_idx:ball_qpos_idx+3] = [x, y, z]
        self.data.qpos[ball_qpos_idx+3:ball_qpos_idx+7] = [1.0, 0.0, 0.0, 0.0]

        # Zero all velocities - ball is stationary
        # Ball velocities start at index 6 (after drone's 6 DOF velocities)
        ball_qvel_idx = 6
        self.data.qvel[ball_qvel_idx:ball_qvel_idx+6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Apply forward kinematics to update state
        mujoco.mj_forward(self.model, self.data)

        self.ball_active = True

        # Verify actual position
        actual_pos = self.data.qpos[ball_qpos_idx:ball_qpos_idx+3]
        print(f"🎾 Ball spawned at ground level:")
        print(f"   Position: x={actual_pos[0]:.1f}, y={actual_pos[1]:.1f}, z={actual_pos[2]:.2f}")

    def check_catch(self):
        """Check if drone touched the ball on the ground."""
        if not self.ball_active:
            return False

        # Get positions
        drone_pos = self.data.qpos[0:3]
        ball_qpos_idx = 7
        ball_pos = self.data.qpos[ball_qpos_idx:ball_qpos_idx+3]

        # Calculate 3D distance between drone and ball
        distance = np.linalg.norm(drone_pos - ball_pos)

        # Ball is touched if drone gets within 0.3m (drone body size + ball size)
        if distance < 0.3:
            return True

        # No "miss" condition - ball stays on ground until touched
        return None  # Still searching

    def apply_control(self):
        """Apply control based on currently pressed keys."""
        # Reset targets
        self.target_thrust = 0.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0

        # Check which keys are currently pressed
        if 'w' in self.keys_pressed:
            self.target_pitch = -0.1
        if 's' in self.keys_pressed:
            self.target_pitch = 0.1
        if 'a' in self.keys_pressed:
            self.target_roll = -0.1
        if 'd' in self.keys_pressed:
            self.target_roll = 0.1
        if 'q' in self.keys_pressed:
            self.target_yaw = -0.1
        if 'e' in self.keys_pressed:
            self.target_yaw = 0.1
        if 'z' in self.keys_pressed:
            self.target_thrust = 1.0
        if 'x' in self.keys_pressed:
            self.target_thrust = -.3

        # Apply damping
        angular_damping = 3.0
        linear_damping = 1.5
        vertical_damping = 2.0

        angular_vel = self.data.qvel[3:6]
        linear_vel = self.data.qvel[0:3]

        if 'w' not in self.keys_pressed and 's' not in self.keys_pressed:
            self.target_pitch -= angular_vel[1] * angular_damping
            self.target_pitch += linear_vel[0] * linear_damping
        if 'a' not in self.keys_pressed and 'd' not in self.keys_pressed:
            self.target_roll -= angular_vel[0] * angular_damping
            self.target_roll += linear_vel[1] * linear_damping
        if 'q' not in self.keys_pressed and 'e' not in self.keys_pressed:
            self.target_yaw -= angular_vel[2] * angular_damping

        if 'z' not in self.keys_pressed and 'x' not in self.keys_pressed:
            self.target_thrust -= linear_vel[2] * vertical_damping

        # Calculate motor outputs
        base_thrust = self.thrust_base
        thrust = base_thrust + self.target_thrust * self.control_gains["thrust"]
        roll = self.target_roll * self.control_gains["roll"]
        pitch = self.target_pitch * self.control_gains["pitch"]
        yaw = self.target_yaw * self.control_gains["yaw"]

        # Quadrotor motor mixing
        motor1 = thrust + roll + pitch + yaw
        motor2 = thrust - roll + pitch - yaw
        motor3 = thrust - roll - pitch + yaw
        motor4 = thrust + roll - pitch - yaw

        motors = [max(0, min(motor, 10.0)) for motor in [motor1, motor2, motor3, motor4]]
        self.data.ctrl[:] = motors

    def apply_ball_drag(self):
        """Apply air resistance to make ball fall slowly."""
        if self.ball_active:
            ball_vel_idx = 6  # Ball velocities start after drone's 6 DOF
            # Apply strong air resistance
            drag_coefficient = 0.95  # Strong drag
            self.data.qvel[ball_vel_idx:ball_vel_idx+3] *= drag_coefficient

    def key_callback(self, keycode):
        """Handle key events."""
        is_release = keycode < 0
        if is_release:
            keycode = -keycode

        if 32 <= keycode <= 126:
            key_char = chr(keycode).lower()
        else:
            key_char = None

        if keycode == 256:  # ESC
            return False
        elif keycode == 32 and not is_release:  # Space
            self.keys_pressed.clear()
            print("⏸️ HOVER")
        elif keycode == 114 and not is_release:  # R - reset
            self.reset_simulation()
            self.score = 0
            self.catches = 0
            self.misses = 0
            self.total_reward = 0.0
            print("🔄 Game reset!")
        elif key_char in ['w', 'a', 's', 'd', 'q', 'e', 'z', 'x']:
            if is_release:
                self.keys_pressed.discard(key_char)
            else:
                self.keys_pressed.add(key_char)

        return True

    def run_game(self):
        """Run the catching game."""
        print("🚀 Starting Drone Catch Game...")
        print("\n" + "="*60)
        print("🎮 CONTROLS:")
        print("  W/A/S/D: Move  |  Q/E: Yaw  |  Z/X: Up/Down")
        print("  SPACE: Hover   |  R: Reset  |  ESC: Exit")
        print("\n🎯 GAME RULES:")
        print("  - Touch balls on the ground with your drone")
        print("  - +10 points for each touch")
        print("  - New ball spawns every 20 seconds")
        print("="*60)

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback
        ) as viewer:

            sim_time = 0
            step_count = 0

            while viewer.is_running():
                step_start = time.time()

                # Apply controls
                self.apply_control()

                # Apply ball drag for slow falling
                self.apply_ball_drag()

                # Step simulation
                mujoco.mj_step(self.model, self.data)
                sim_time += self.model.opt.timestep

                # Check if it's time to spawn a ball
                if sim_time - self.last_spawn_time >= self.ball_spawn_interval and not self.ball_active:
                    self.spawn_ball()
                    self.last_spawn_time = sim_time

                # Check for touch
                if self.ball_active:
                    result = self.check_catch()
                    if result == True:  # Touched!
                        self.catches += 1
                        self.score += 10
                        self.total_reward += 10.0
                        self.ball_active = False
                        print(f"✅ TOUCHED! +10 points | Score: {self.score} | Touches: {self.catches}")

                # Update viewer
                viewer.sync()

                # Print status occasionally
                step_count += 1
                if step_count % 1000 == 0:
                    drone_pos = self.data.qpos[:3]
                    ball_qpos_idx = 7
                    ball_pos = self.data.qpos[ball_qpos_idx:ball_qpos_idx+3]

                    status = f"📊 Score: {self.score} | Catches: {self.catches} | Misses: {self.misses} | "
                    status += f"Drone: ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f})"
                    if self.ball_active:
                        status += f" | Ball: ({ball_pos[0]:.1f}, {ball_pos[1]:.1f}, {ball_pos[2]:.1f})"
                    print(status)

                # Control simulation rate
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        print(f"\n🏁 Game Over!")
        print(f"Final Score: {self.score}")
        print(f"Catches: {self.catches} | Misses: {self.misses}")
        print(f"Total Reward: {self.total_reward}")


def main():
    """Main function."""
    try:
        game = DroneCatchGame("drone_model.xml")
        game.run_game()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

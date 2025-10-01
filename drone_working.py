#!/usr/bin/env python3
"""
Working Drone Simulator that actually responds to controls
Uses MuJoCo's built-in control system properly
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import os
from datetime import datetime


class WorkingDrone:
    def __init__(self, model_path: str = "drone_model.xml"):
        """Initialize working drone simulator."""
        print("🚁 Loading Working Drone Simulator...")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Control parameters
        self.thrust_base = 3.2  # Higher base thrust
        self.control_gains = {"roll": 2.0, "pitch": 2.0, "yaw": 1.5, "thrust": 3.0}

        # Control state - using continuous control
        self.target_thrust = 0.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0

        # Key state tracking
        self.keys_pressed = set()

        # Recording state
        self.is_recording = False
        self.frame_buffer = []
        self.recording_timestamp = None

        # Initialize drone
        self.reset_drone()

        print("✅ Working drone ready!")

    def reset_drone(self):
        """Reset drone to starting position."""
        self.data.qpos[0:3] = [0.0, 0.0, 2.0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        print("🔄 Drone reset to starting position")

    def start_recording(self):
        """Start recording simulation data."""
        if self.is_recording:
            print("⚠️ Already recording!")
            return

        self.is_recording = True
        self.frame_buffer = []
        self.recording_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"🎥 Recording started! Data will be saved with timestamp: {self.recording_timestamp}")
        print("⚠️ Note: Due to WSL/OpenGL limitations, recording telemetry data only (not video)")

    def stop_recording(self):
        """Stop recording and save data."""
        if not self.is_recording:
            print("⚠️ Not currently recording!")
            return

        self.is_recording = False

        # Save telemetry data
        if len(self.frame_buffer) > 0:
            os.makedirs("recordings", exist_ok=True)
            filename = f"recordings/flight_data_{self.recording_timestamp}.csv"

            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'pos_x', 'pos_y', 'pos_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z',
                               'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                               'motor1', 'motor2', 'motor3', 'motor4'])
                writer.writerows(self.frame_buffer)

            print(f"⏹️ Recording stopped! Saved {len(self.frame_buffer)} frames to {filename}")
        else:
            print("⏹️ Recording stopped! No data recorded")

        self.frame_buffer = []

    def capture_frame(self):
        """Capture telemetry data frame."""
        if not self.is_recording:
            return

        # Record telemetry data
        import time
        frame_data = [
            time.time(),
            self.data.qpos[0], self.data.qpos[1], self.data.qpos[2],  # position
            self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6],  # quaternion
            self.data.qvel[0], self.data.qvel[1], self.data.qvel[2],  # linear velocity
            self.data.qvel[3], self.data.qvel[4], self.data.qvel[5],  # angular velocity
            self.data.ctrl[0], self.data.ctrl[1], self.data.ctrl[2], self.data.ctrl[3]  # motor commands
        ]
        self.frame_buffer.append(frame_data)

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

        # Apply damping - counter both angular and linear velocity
        angular_damping = 3.0  # Much stronger angular damping
        linear_damping = 1.5   # Stronger linear damping
        vertical_damping = 2.0 # Strong vertical damping

        angular_vel = self.data.qvel[3:6]  # Angular velocity (roll, pitch, yaw rates)
        linear_vel = self.data.qvel[0:3]   # Linear velocity (x, y, z movement)

        # If no directional keys pressed, apply strong damping to stop rotation
        if 'w' not in self.keys_pressed and 's' not in self.keys_pressed:
            self.target_pitch -= angular_vel[1] * angular_damping
            # Counter forward/backward movement
            self.target_pitch += linear_vel[0] * linear_damping
        if 'a' not in self.keys_pressed and 'd' not in self.keys_pressed:
            self.target_roll -= angular_vel[0] * angular_damping
            # Counter left/right movement
            self.target_roll += linear_vel[1] * linear_damping
        if 'q' not in self.keys_pressed and 'e' not in self.keys_pressed:
            self.target_yaw -= angular_vel[2] * angular_damping

        # Counter vertical velocity when not pressing Z or X
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

        # Clamp motor values
        motors = [max(0, min(motor, 10.0)) for motor in [motor1, motor2, motor3, motor4]]
        self.data.ctrl[:] = motors

        # Print active controls (disabled - too verbose)
        # if any([self.target_roll, self.target_pitch, self.target_yaw, self.target_thrust]):
        #     controls = []
        #     if self.target_pitch < 0: controls.append("Forward")
        #     if self.target_pitch > 0: controls.append("Backward")
        #     if self.target_roll < 0: controls.append("Left")
        #     if self.target_roll > 0: controls.append("Right")
        #     if self.target_yaw < 0: controls.append("Yaw-L")
        #     if self.target_yaw > 0: controls.append("Yaw-R")
        #     if self.target_thrust > 0: controls.append("Up")
        #     if self.target_thrust < 0: controls.append("Down")
        #     if controls:
        #         print(f"🎮 {', '.join(controls)} | Motors: [{motors[0]:.1f}, {motors[1]:.1f}, {motors[2]:.1f}, {motors[3]:.1f}]")

    def key_callback(self, keycode):
        """Handle key events."""
        # Check if this is a key release event (negative keycode)
        is_release = keycode < 0
        if is_release:
            keycode = -keycode

        # Convert keycode to character
        if 32 <= keycode <= 126:
            key_char = chr(keycode).lower()
        else:
            key_char = None

        if keycode == 256:  # ESC
            return False
        elif keycode == 32 and not is_release:  # Space - emergency hover
            self.keys_pressed.clear()
            print("⏸️ EMERGENCY HOVER - All controls cleared")
        elif keycode == 114 and not is_release:  # R - reset
            self.reset_drone()
            self.keys_pressed.clear()
        elif keycode == 86 and not is_release:  # V - toggle recording
            if self.is_recording:
                self.stop_recording()
            else:
                self.start_recording()
        elif key_char in ['w', 'a', 's', 'd', 'q', 'e', 'z', 'x']:
            if is_release:
                # Remove key from pressed set when released
                self.keys_pressed.discard(key_char)
            else:
                # Add key to pressed set when pressed
                self.keys_pressed.add(key_char)
                action_map = {
                    'w': '⬆️ Forward', 's': '⬇️ Backward',
                    'a': '⬅️ Left', 'd': '➡️ Right',
                    'q': '↶ Yaw Left', 'e': '↷ Yaw Right',
                    'z': '🔺 Up', 'x': '🔻 Down'
                }
                print(f"{action_map.get(key_char, key_char)} activated")

        return True

    def run_simulation(self):
        """Run the working simulation."""
        print("🚀 Starting Working Drone Simulation...")
        print("\n" + "="*60)
        print("🎮 WORKING CONTROLS (hold keys down!):")
        print("  W: Forward    |  S: Backward")
        print("  A: Left       |  D: Right")
        print("  Q: Yaw Left   |  E: Yaw Right")
        print("  Z: Up         |  X: Down")
        print("  SPACE: Emergency Hover (stops all movement)")
        print("  R: Reset drone position")
        print("  V: Start/Stop Recording")
        print("  ESC: Exit")
        print("\n📹 Camera Controls (MuJoCo built-in):")
        print("  [ ]: Previous camera")
        print("  ] : Next camera")
        print("  Mouse: Rotate/Pan/Zoom view")
        print("="*60)
        print("\n🎯 IMPORTANT: HOLD keys down for movement!")
        print("💡 TIP: Use [ and ] to switch between camera views")
        print("🎬 TIP: Press V to start/stop recording your flight!")

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback
        ) as viewer:

            step_count = 0
            last_print = 0

            while viewer.is_running():
                step_start = time.time()

                # Apply controls based on currently pressed keys
                self.apply_control()

                # Step simulation
                mujoco.mj_step(self.model, self.data)

                # Capture frame if recording
                self.capture_frame()

                # Update viewer
                viewer.sync()

                # Print status occasionally
                step_count += 1
                if step_count - last_print > 500:
                    pos = self.data.qpos[:3]
                    height = pos[2]
                    status = "FLYING" if height > 0.5 else "LANDED"
                    print(f"📍 Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | Status: {status}")
                    if len(self.keys_pressed) > 0:
                        print(f"🎮 Active keys: {', '.join(sorted(self.keys_pressed))}")
                    last_print = step_count

                # Control simulation rate
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        # Stop recording if still active
        if self.is_recording:
            self.stop_recording()

        print("🛑 Simulation ended.")


def main():
    """Main function."""
    try:
        drone = WorkingDrone("drone_model.xml")
        drone.run_simulation()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Record a simple flight: takeoff (Z for 1 second) then hover for 5 seconds
"""

import mujoco
import numpy as np
import cv2
import os

def record_simple_flight():
    """Record simple takeoff and hover flight."""
    print("🎬 Recording simple flight: takeoff + hover...")

    # Set up for offscreen rendering
    os.environ['MUJOCO_GL'] = 'osmesa'

    try:
        model = mujoco.MjModel.from_xml_path("drone_model.xml")
        data = mujoco.MjData(model)

        # Create renderer
        renderer = mujoco.Renderer(model, height=480, width=640)

        # Initialize drone
        data.qpos[0:3] = [0.0, 0.0, 2.0]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        # Set up video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs("recordings", exist_ok=True)

        cameras = ["external_camera", "drone_camera", "fpv_camera", "top_camera"]
        writers = {}

        for cam in cameras:
            writers[cam] = cv2.VideoWriter(f"recordings/simple_{cam}.mp4", fourcc, 30.0, (640, 480))

        print("🎥 Recording started...")

        # Control parameters (matching drone_working.py)
        thrust_base = 3.2
        control_gains = {"roll": 2.0, "pitch": 2.0, "yaw": 1.5, "thrust": 3.0}

        # Render every 5th simulation step for speed (still 30fps video)
        render_every = 5
        total_frames = 0

        # Phase 1: Takeoff (Z key - thrust up) for 1 second (200 steps)
        print("Phase 1: Takeoff (1 second)...")
        for step in range(200):
            target_thrust = 1.0  # Z key pressed

            # Calculate motor outputs
            thrust = thrust_base + target_thrust * control_gains["thrust"]
            motors = [thrust, thrust, thrust, thrust]
            motors = [max(0, min(m, 10.0)) for m in motors]
            data.ctrl[:] = motors

            # Step simulation
            mujoco.mj_step(model, data)

            # Only render every Nth frame
            if step % render_every == 0:
                for cam in cameras:
                    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                    renderer.update_scene(data, cam_id)
                    frame = renderer.render()
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writers[cam].write(frame_bgr)
                total_frames += 1
                if total_frames % 10 == 0:
                    print(f"  Rendered {total_frames} frames...")

        # Phase 2: Hover for 5 seconds (1000 steps)
        print("Phase 2: Hover (5 seconds)...")

        # Damping parameters
        angular_damping = 3.0
        linear_damping = 1.5
        vertical_damping = 2.0

        for step in range(1000):
            # Apply damping to stop movement
            angular_vel = data.qvel[3:6]
            linear_vel = data.qvel[0:3]

            target_roll = -angular_vel[0] * angular_damping + linear_vel[1] * linear_damping
            target_pitch = -angular_vel[1] * angular_damping + linear_vel[0] * linear_damping
            target_yaw = -angular_vel[2] * angular_damping
            target_thrust = -linear_vel[2] * vertical_damping

            # Calculate motor outputs
            thrust = thrust_base + target_thrust * control_gains["thrust"]
            roll = target_roll * control_gains["roll"]
            pitch = target_pitch * control_gains["pitch"]
            yaw = target_yaw * control_gains["yaw"]

            # Quadrotor motor mixing
            motor1 = thrust + roll + pitch + yaw
            motor2 = thrust - roll + pitch - yaw
            motor3 = thrust - roll - pitch + yaw
            motor4 = thrust + roll - pitch - yaw

            motors = [max(0, min(m, 10.0)) for m in [motor1, motor2, motor3, motor4]]
            data.ctrl[:] = motors

            # Step simulation
            mujoco.mj_step(model, data)

            # Only render every Nth frame
            if step % render_every == 0:
                for cam in cameras:
                    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                    renderer.update_scene(data, cam_id)
                    frame = renderer.render()
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writers[cam].write(frame_bgr)
                total_frames += 1
                if total_frames % 10 == 0:
                    print(f"  Rendered {total_frames} frames...")

        # IMPORTANT: Close video writers properly
        print("Finalizing videos...")
        for writer in writers.values():
            writer.release()

        print(f"✅ Recording complete! Rendered {total_frames} frames")
        print("📁 Videos saved in recordings/ folder:")
        for cam in cameras:
            print(f"   - simple_{cam}.mp4")

    except Exception as e:
        print(f"❌ Recording failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    record_simple_flight()

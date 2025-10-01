#!/usr/bin/env python3
"""
Record drone flight to video files for later viewing
"""

import mujoco
import numpy as np
import cv2
import os

def record_flight_demo():
    """Record a demonstration flight."""
    print("🎬 Recording drone flight demonstration...")

    # Set up for offscreen rendering
    os.environ['MUJOCO_GL'] = 'osmesa'

    try:
        model = mujoco.MjModel.from_xml_path("drone_model.xml")
        data = mujoco.MjData(model)

        # Fix the renderer dimensions
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
            writers[cam] = cv2.VideoWriter(f"recordings/demo_{cam}.mp4", fourcc, 30.0, (640, 480))

        print("🎥 Recording started...")

        # Demonstrate flight sequence
        flight_sequence = [
            # (duration, roll, pitch, yaw, thrust)
            (100, 0, 0, 0, 0.5),      # Take off
            (50, 0, 0, 0, 0),         # Hover
            (100, 0, -0.3, 0, 0),     # Move forward
            (50, 0, 0, 0, 0),         # Hover
            (100, -0.3, 0, 0, 0),     # Move left
            (50, 0, 0, 0, 0),         # Hover
            (100, 0, 0, 0.3, 0),      # Yaw right
            (50, 0, 0, 0, 0),         # Hover
            (100, 0, 0.3, 0, 0),      # Move backward
            (50, 0, 0, 0, 0),         # Hover
            (100, 0, 0, 0, -0.5),     # Descend
        ]

        step = 0
        for duration, roll, pitch, yaw, thrust in flight_sequence:
            print(f"Flight phase: roll={roll}, pitch={pitch}, yaw={yaw}, thrust={thrust}")

            for _ in range(duration):
                # Apply controls
                base_thrust = 2.5
                total_thrust = base_thrust + thrust * 1.0

                # Motor mixing
                motor1 = total_thrust + roll*0.5 + pitch*0.5 + yaw*0.3
                motor2 = total_thrust - roll*0.5 + pitch*0.5 - yaw*0.3
                motor3 = total_thrust - roll*0.5 - pitch*0.5 + yaw*0.3
                motor4 = total_thrust + roll*0.5 - pitch*0.5 - yaw*0.3

                motors = [max(0, min(motor, 5.0)) for motor in [motor1, motor2, motor3, motor4]]
                data.ctrl[:] = motors

                # Step simulation
                mujoco.mj_step(model, data)

                # Capture and record frames
                for cam in cameras:
                    try:
                        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                        renderer.update_scene(data, cam_id)
                        frame = renderer.render()
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writers[cam].write(frame_bgr)
                    except:
                        pass

                step += 1
                if step % 100 == 0:
                    pos = data.qpos[:3]
                    print(f"Step {step}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        # Close video writers
        for writer in writers.values():
            writer.release()

        print("✅ Recording complete!")
        print("📁 Videos saved in recordings/ folder:")
        for cam in cameras:
            print(f"   - demo_{cam}.mp4")

    except Exception as e:
        print(f"❌ Recording failed: {e}")
        print("This requires proper MuJoCo rendering support.")

if __name__ == "__main__":
    record_flight_demo()
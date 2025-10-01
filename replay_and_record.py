#!/usr/bin/env python3
"""
Replay recorded telemetry data and render videos from all camera angles.
This script reads CSV telemetry data from interactive flights and renders video.
"""

import mujoco
import numpy as np
import cv2
import os
import csv
import sys

def replay_recording(csv_file):
    """Replay a recorded flight and generate videos."""
    print(f"🎬 Replaying recording: {csv_file}")

    # Set up for offscreen rendering
    os.environ['MUJOCO_GL'] = 'osmesa'

    try:
        # Load model and data
        model = mujoco.MjModel.from_xml_path("drone_model.xml")
        data = mujoco.MjData(model)

        # Create renderer
        renderer = mujoco.Renderer(model, height=480, width=640)

        # Load telemetry data
        frames = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames.append(row)

        print(f"📊 Loaded {len(frames)} frames")

        # Set up video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs("recordings", exist_ok=True)

        # Extract timestamp from filename
        base_name = os.path.basename(csv_file).replace('flight_data_', '').replace('.csv', '')

        cameras = ["external_camera", "drone_camera", "fpv_camera", "top_camera"]
        writers = {}

        for cam in cameras:
            filename = f"recordings/replay_{cam}_{base_name}.mp4"
            writers[cam] = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
            print(f"📹 Will save: {filename}")

        print("🎥 Rendering videos...")

        # Replay each frame
        for i, frame in enumerate(frames):
            # Set drone state from telemetry
            data.qpos[0] = float(frame['pos_x'])
            data.qpos[1] = float(frame['pos_y'])
            data.qpos[2] = float(frame['pos_z'])
            data.qpos[3] = float(frame['quat_w'])
            data.qpos[4] = float(frame['quat_x'])
            data.qpos[5] = float(frame['quat_y'])
            data.qpos[6] = float(frame['quat_z'])

            data.qvel[0] = float(frame['vel_x'])
            data.qvel[1] = float(frame['vel_y'])
            data.qvel[2] = float(frame['vel_z'])
            data.qvel[3] = float(frame['ang_vel_x'])
            data.qvel[4] = float(frame['ang_vel_y'])
            data.qvel[5] = float(frame['ang_vel_z'])

            data.ctrl[0] = float(frame['motor1'])
            data.ctrl[1] = float(frame['motor2'])
            data.ctrl[2] = float(frame['motor3'])
            data.ctrl[3] = float(frame['motor4'])

            # Update simulation
            mujoco.mj_forward(model, data)

            # Render from each camera
            for cam in cameras:
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                renderer.update_scene(data, cam_id)
                frame_rgb = renderer.render()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writers[cam].write(frame_bgr)

            # Progress update
            if (i + 1) % 100 == 0:
                print(f"  Rendered {i + 1}/{len(frames)} frames...")

        # Close video writers
        for writer in writers.values():
            writer.release()

        print("✅ Video rendering complete!")
        print("📁 Videos saved in recordings/ folder:")
        for cam in cameras:
            print(f"   - replay_{cam}_{base_name}.mp4")

    except Exception as e:
        print(f"❌ Replay failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 replay_and_record.py <telemetry_csv_file>")
        print("\nExample: python3 replay_and_record.py recordings/flight_data_20250930_203124.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)

    replay_recording(csv_file)

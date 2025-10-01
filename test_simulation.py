#!/usr/bin/env python3
"""
Quick test of the drone simulation without GUI to check for WSL issues.
"""

import mujoco
import numpy as np
import cv2
import os

def test_simulation():
    """Test basic simulation functionality."""
    print("Testing drone simulation...")

    # Load model
    model = mujoco.MjModel.from_xml_path("drone_model.xml")
    data = mujoco.MjData(model)

    print(f"Model loaded: {model.nbody} bodies, {model.nq} DOF")

    # Test renderer
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
        print("Renderer initialized successfully")

        # Try to render a frame
        renderer.update_scene(data)
        frame = renderer.render()
        print(f"Frame captured: {frame.shape}")

        # Try different cameras
        for i, cam_name in enumerate(["external_camera", "drone_camera", "fpv_camera", "top_camera"]):
            try:
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                renderer.update_scene(data, cam_id)
                frame = renderer.render()
                print(f"Camera '{cam_name}': OK ({frame.shape})")

                # Save a test frame
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"test_{cam_name}.png", frame_bgr)

            except Exception as e:
                print(f"Camera '{cam_name}': Error - {e}")

        print("\nTest frames saved as test_*.png")

    except Exception as e:
        print(f"Renderer error: {e}")
        return False

    # Test simulation steps
    print("\nTesting simulation steps...")
    for step in range(10):
        data.ctrl[:] = [2.5, 2.5, 2.5, 2.5]  # Base thrust
        mujoco.mj_step(model, data)

    pos = data.qpos[:3]
    print(f"Final drone position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

    print("✅ Basic simulation test passed!")
    return True

if __name__ == "__main__":
    test_simulation()
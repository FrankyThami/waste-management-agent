"""
adaptive_validation/locate_targets.py

Day 1: find "yellow paper" and "red dustbin" in 2D, then turn each
detection into a 3D position using the D435i's depth stream.

Run from the project root, venv active:
    python -m adaptive_validation.locate_targets
"""

import pyrealsense2 as rs
import numpy as np
from PIL import Image

from perception.gemini_classifier import classify_image
from prompts.prompts_v1 import PROMPT_LOCATE_TARGETS
from adaptive_validation.geometry import locate_object


def capture_aligned_frames():
    """Grab one aligned color+depth frame pair. Depth is aligned to the
    color sensor's viewpoint, so pixel (x, y) means the same real-world
    point in both frames."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    for _ in range(5):  # let auto-exposure settle
        frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    return pipeline, aligned.get_color_frame(), aligned.get_depth_frame()


def main():
    print("Capturing aligned color + depth from D435i...")
    pipeline, color_frame, depth_frame = capture_aligned_frames()

    try:
        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = Image.fromarray(color_image[:, :, ::-1])  # BGR -> RGB

        print("Asking Gemini Robotics ER 2 to find the yellow paper and red dustbin...")
        detections = classify_image(rgb_image, prompt=PROMPT_LOCATE_TARGETS)

        print("\nResults:")
        for det in detections:
            target = det.get("target", "unknown")
            point = det.get("point")

            if not det.get("found", False) or point is None:
                print(f"  {target}: not found")
                continue

            result = locate_object(point, color_frame, depth_frame)
            if result["valid"]:
                print(f"  {target}: pixel={result['pixel']}, "
                      f"camera_xyz_m={result['camera_xyz_m']}, "
                      f"robot_xyz_m={result['robot_xyz_m']}")
            else:
                print(f"  {target}: found in 2D but 3D lookup failed - {result['reason']}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
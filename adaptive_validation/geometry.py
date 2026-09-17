"""
adaptive_validation/geometry.py

Turns a 2D point from Gemini (normalized [y, x], 0-1000) into a 3D
position in the robot's coordinate frame. This is the standalone
logic behind the locate_object() tool from the project spec.

Camera is a FIXED OVERHEAD mount (not wrist-mounted), so this is a
one-time static "eye-to-hand" transform, not dynamic hand-eye
calibration - see config/camera_to_base_transform.json.
"""

import json
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

TRANSFORM_PATH = Path(__file__).resolve().parent.parent / "config" / "camera_to_base_transform.json"

_warned_mock = False


def load_camera_to_base_transform():
    """Load the eye-to-hand calibration: a 3x3 rotation matrix and a
    translation vector, converted here from mm (as stored in the file)
    to meters (what pyrealsense2 works in)."""
    global _warned_mock
    with open(TRANSFORM_PATH, "r") as f:
        data = json.load(f)

    rotation = np.array(data["rotation_matrix"], dtype=float)
    translation_m = np.array(data["translation_vector_mm"], dtype=float) / 1000.0

    if data.get("status", "").startswith("MOCK") and not _warned_mock:
        print(f"[locate_object] NOTE: {TRANSFORM_PATH.name} is still the mock "
              f"identity transform - 3D positions below are camera-frame "
              f"coordinates, not real robot-frame ones yet.")
        _warned_mock = True

    return rotation, translation_m


def get_depth_at_pixel(depth_frame, x_px: int, y_px: int, patch: int = 5):
    """Robust depth lookup: median of a small patch around the pixel,
    ignoring 0 (invalid/no-return) readings. Returns meters, or None if
    every pixel in the patch was invalid (common on shiny/transparent
    items - depth gaps)."""
    half = patch // 2
    values = []
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            px, py = x_px + dx, y_px + dy
            if 0 <= px < depth_frame.get_width() and 0 <= py < depth_frame.get_height():
                d = depth_frame.get_distance(px, py)
                if d > 0:
                    values.append(d)
    return float(np.median(values)) if values else None


def locate_object(norm_point_yx, color_frame, depth_frame) -> dict:
    """
    norm_point_yx: [y, x] from Gemini, normalized 0-1000.
    color_frame / depth_frame: an ALIGNED pair from the same capture
    (depth_frame must already be the output of rs.align, not raw).

    Returns: {"valid": bool, ...3D data... } or {"valid": False, "reason": str}
    """
    y_norm, x_norm = norm_point_yx
    width, height = color_frame.get_width(), color_frame.get_height()
    x_px = int(x_norm / 1000 * width)
    y_px = int(y_norm / 1000 * height)

    depth_m = get_depth_at_pixel(depth_frame, x_px, y_px)
    if depth_m is None:
        return {"valid": False, "reason": f"no valid depth near pixel ({x_px}, {y_px})"}

    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    camera_xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [x_px, y_px], depth_m)

    rotation, translation = load_camera_to_base_transform()
    robot_xyz = rotation @ np.array(camera_xyz) + translation

    return {
        "valid": True,
        "pixel": [x_px, y_px],
        "camera_xyz_m": [round(v, 4) for v in camera_xyz],
        "robot_xyz_m": [round(v, 4) for v in robot_xyz.tolist()],
    }
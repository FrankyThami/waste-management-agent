"""
frame_transform.py - the camera -> robot-base transform (Stage 2.3b).

Robot base frame (right-handed, metres):
  origin O : on the tabletop, at the centre of the (future) robot base
  z        : UP, away from the table (the table-plane normal), so z = 0 is the tabletop
  x        : toward the image bottom, i.e. away from the robot across the table
  y        : z cross x -> toward the image right (the robot's left)

A point converts like this:   p_base = R @ p_camera + t

The numbers live in config/camera_to_base_transform.json. Right now that file is
PROVISIONAL (built from the table plane and one rough distance). A real calibration
later just overwrites the file - no code has to change.
"""

import json
import os
import shutil
from datetime import datetime

import numpy as np
import pyrealsense2 as rs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRANSFORM_PATH = os.path.join(PROJECT_ROOT, "config", "camera_to_base_transform.json")

_loaded = None   # (rotation, translation), read from the file the first time it is needed


def table_point_at_pixel(plane, intrinsics, px, py):
    """Camera-frame point where the camera ray through pixel (px, py) meets the table plane."""
    ray = np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [float(px), float(py)], 1.0))
    scale = plane.d / float(np.dot(np.array(plane.normal), ray))
    return scale * ray


def build_provisional_transform(plane, intrinsics, image_width, robot_distance_m):
    """
    Builds R and t from the table plane plus ONE rough number:
    robot_distance_m = distance from the robot base centre to the TOP EDGE of the camera
    image, along the table. The robot is assumed to sit on the image's vertical centre line.
    """
    z_axis = np.array(plane.normal, dtype=float)             # table normal, pointing up toward the camera
    z_axis /= np.linalg.norm(z_axis)

    image_down = np.array([0.0, 1.0, 0.0])                   # camera y axis = toward the image bottom
    x_axis = image_down - np.dot(image_down, z_axis) * z_axis    # flattened into the table plane
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)                        # right-hand rule
    rotation = np.vstack([x_axis, y_axis, z_axis])           # rows = robot axes, in camera coordinates

    top_centre = table_point_at_pixel(plane, intrinsics, image_width / 2, 0)
    origin_cam = top_centre - robot_distance_m * x_axis      # the robot base, in camera coordinates
    translation = -rotation @ origin_cam                     # so that p_base = R @ p_cam + t

    return {
        "rotation_matrix": rotation.tolist(),
        "translation_vector_m": [float(v) for v in translation],
        "robot_distance_m": float(robot_distance_m),
    }


def save_transform(result, status, note):
    """Writes the transform to config/camera_to_base_transform.json (backs up the old file once)."""
    global _loaded
    if os.path.exists(TRANSFORM_PATH):
        backup_path = TRANSFORM_PATH.replace(".json", ".backup.json")
        if not os.path.exists(backup_path):
            shutil.copy(TRANSFORM_PATH, backup_path)
            print(f"Old file backed up as {backup_path}")

    data = {
        "status": status,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "convention": "p_base = rotation_matrix @ p_camera + translation_vector_m ; metres ; "
                      "base z is up, z = 0 is the tabletop",
        "note": note,
        **result,
    }
    with open(TRANSFORM_PATH, "w") as f:
        json.dump(data, f, indent=2)
    _loaded = None   # make sure the next conversion reads the new file


def _load():
    global _loaded
    if _loaded is None:
        with open(TRANSFORM_PATH, "r") as f:
            data = json.load(f)
        if "translation_vector_m" not in data:
            raise ValueError(
                "config/camera_to_base_transform.json is still the old placeholder (identity, millimetres). "
                "Run:  python scripts/set_provisional_frame.py"
            )
        _loaded = (np.array(data["rotation_matrix"], dtype=float),
                   np.array(data["translation_vector_m"], dtype=float))
    return _loaded


def camera_to_base_point(point):
    """Converts one camera-frame point (x, y, z) in metres into the robot-base frame."""
    rotation, translation = _load()
    base = rotation @ np.array(point, dtype=float) + translation
    return tuple(round(float(v), 4) for v in base)
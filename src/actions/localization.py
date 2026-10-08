"""
localization.py - turns Scene pixels into real 3D positions (Stage 2.3).

Camera frame convention (RealSense): x = right, y = down, z = distance from the
camera. All positions are in metres.

Objects: the point is the middle of the object's TOP surface (where the gripper
         will touch it), using the median depth of the central part of its box.
Bins:    the point is the middle of the bin at RIM height, using the closest
         surface inside the bin's box. The rim is where an item is released.
"""

import numpy as np
import pyrealsense2 as rs

from src.actions.camera import CapturedFrame
from src.context.scene import Scene
from src.actions.frame_transform import camera_to_base_point

CENTRAL_FRACTION = 0.3      # objects: use the middle 30% of the box (in width and height)
MIN_VALID_FRACTION = 0.3    # at least 30% of the sampled pixels must have a real depth reading
BIN_RIM_PERCENTILE = 0.5     # bins: the rim = the closest 10% of valid depth readings in the box
MIN_TABLE_PIXELS = 5000     # table reference: need at least this many bare-table readings

def camera_to_base(point):
    """
    Converts a point from the camera frame into the robot-base frame, using the
    transform saved in config/camera_to_base_transform.json (see frame_transform.py).
    """
    return camera_to_base_point(point)


def _central_region(box_px, fraction):
    """The middle `fraction` of a box, as (x1, y1, x2, y2) pixel limits."""
    x1, y1, x2, y2 = box_px
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half_w = max(2, int((x2 - x1) * fraction / 2))
    half_h = max(2, int((y2 - y1) * fraction / 2))
    return cx - half_w, cy - half_h, cx + half_w + 1, cy + half_h + 1


def _valid_depths(depth_m, region):
    """All real depth readings (not 0) inside a region, and how many pixels the region had."""
    x1, y1, x2, y2 = region
    patch = depth_m[max(0, y1):y2, max(0, x1):x2]
    return patch[patch > 0], patch.size


def _object_depth(depth_m, box_px):
    """Median depth of the central part of an object's box, or None if unreliable."""
    valid, total = _valid_depths(depth_m, _central_region(box_px, CENTRAL_FRACTION))
    if total == 0 or len(valid) / total < MIN_VALID_FRACTION:
        return None
    return float(np.median(valid))


def _bin_rim_depth(depth_m, box_px):
    """Depth of the closest surface in a bin's box (its rim), or None if unreliable."""
    valid, total = _valid_depths(depth_m, box_px)
    if total == 0 or len(valid) / total < MIN_VALID_FRACTION:
        return None
    return float(np.percentile(valid, BIN_RIM_PERCENTILE))


def _deproject(intrinsics, px, py, depth):
    """Pixel + depth -> 3D point (x, y, z) in metres, in the camera frame."""
    x, y, z = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth)
    return round(x, 4), round(y, 4), round(z, 4)


def localize_scene(scene: Scene, frame: CapturedFrame) -> None:
    """
    Fills in position_camera, position_base and depth_valid for every object
    and bin in the Scene (changes the Scene in place). Items with no reliable
    depth keep depth_valid = False, with a printed note.
    """
    for obj in scene.objects:
        depth = _object_depth(frame.depth_m, obj.box_px)
        if depth is None:
            print(f"[localization] {obj.object_id} ({obj.waste_class}): no reliable depth reading")
            continue
        point = _deproject(frame.intrinsics, *obj.center_px, depth)
        obj.position_camera = point
        obj.position_base = camera_to_base(point)
        obj.depth_valid = True

    for b in scene.bins:
        depth = _bin_rim_depth(frame.depth_m, b.box_px)
        if depth is None:
            print(f"[localization] {b.bin_id} ({b.colour} bin): no reliable depth reading")
            continue
        point = _deproject(frame.intrinsics, *b.center_px, depth)
        b.position_camera = point
        b.position_base = camera_to_base(point)
        b.depth_valid = True

def table_depth_samples(depth_m: np.ndarray, scene: Scene) -> np.ndarray:
    """
    All valid depth readings (metres) OUTSIDE every detected object and bin box.
    Everything outside the boxes is the bare table surface.
    """
    on_table = np.ones(depth_m.shape, dtype=bool)
    for item in [*scene.objects, *scene.bins]:
        x1, y1, x2, y2 = item.box_px
        on_table[y1:y2 + 1, x1:x2 + 1] = False
    samples = depth_m[on_table]
    return samples[samples > 0]


def estimate_table_depth(depth_m: np.ndarray, scene: Scene) -> float | None:
    """Median depth (metres) of the bare table, or None if too little table is visible."""
    samples = table_depth_samples(depth_m, scene)
    if samples.size < MIN_TABLE_PIXELS:
        return None
    return float(np.median(samples))
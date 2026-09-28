"""
agent/tools.py

The 4 tools the agent can call. Built one at a time on Day 1.
So far: observe_scene() (Step 4), attempt_pick() (Step 5), attempt_place() (Step 6).
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from perception.gemini_classifier import classify_image, ClassificationError
from offline_reasoning.results_store import normalize_class
from agent.schemas import DetectedObject, ToolResult
from agent.hal import check_reachable, simulate_move_to, simulate_grasp, simulate_release, simulate_stop, get_bin_position


# ---------------------------------------------------------------------
# observe_scene() - Step 4
# ---------------------------------------------------------------------

def _box_center_pixel(box_2d, width, height):
    """Gemini's box_2d is [ymin, xmin, ymax, xmax], normalized 0-1000."""
    ymin, xmin, ymax, xmax = box_2d
    cx = int((xmin + xmax) / 2 / 1000 * width)
    cy = int((ymin + ymax) / 2 / 1000 * height)
    return cx, cy


def _apply_camera_to_base_transform(camera_point):
    """
    Converts a point from the camera frame into the robot base frame.

    Currently an identity mapping, matching the placeholder transform in
    config/camera_to_base_transform.json. Once the real static extrinsic
    calibration is measured, this one function is where that matrix gets
    applied - nothing else in the agent needs to change.
    """
    return camera_point


def _pixel_to_position(depth_frame, intrinsics, px, py):
    """Combine a pixel + its depth reading into a {x, y, z} point, in metres."""
    depth_m = depth_frame.get_distance(px, py)
    if depth_m == 0:
        return None  # no valid depth here - common on shiny/transparent items
    x, y, z = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_m)
    return _apply_camera_to_base_transform({"x": x, "y": y, "z": z})


def observe_scene():
    """
    Captures one frame from the RealSense D435i, asks Gemini what waste
    objects are in it, and returns a list of DetectedObject dicts - each
    with a class, a 3D position, and a confidence.

    Detections with no valid depth reading are skipped rather than
    guessed at, with a note printed so nothing fails silently.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        for _ in range(15):  # let auto-exposure settle before the real capture
            pipeline.wait_for_frames()

        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense did not return a valid frame.")

        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())
        height, width = color_image.shape[:2]
    finally:
        pipeline.stop()

    rgb_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    try:
        detections = classify_image(rgb_image)
    except ClassificationError as e:
        print(f"[observe_scene] Gemini call failed: {e}")
        return []

    objects = []
    for i, det in enumerate(detections):
        if "material" not in det:
            print(f"[observe_scene] Skipping object {i}: no 'material' field in Gemini's response ({det})")
            continue
        try:
            waste_class = normalize_class(det["material"]).lower()
        except ValueError:
            print(f"[observe_scene] Skipping unrecognized class '{det['material']}'")
            continue

        px, py = _box_center_pixel(det["box_2d"], width, height)
        position = _pixel_to_position(depth_frame, intrinsics, px, py)
        if position is None:
            print(f"[observe_scene] Skipping object {i}: no valid depth reading")
            continue

        obj: DetectedObject = {
            "id": f"obj_{i}",
            "waste_class": waste_class,
            "position": position,
            "confidence": det.get("confidence", 1.0),  # Gemini isn't asked for this yet - placeholder
        }
        objects.append(obj)

    return objects


# ---------------------------------------------------------------------
# attempt_pick() - Step 5
# ---------------------------------------------------------------------

def _localize(target):
    """Step 1: read the object's position (already computed by observe_scene)."""
    return target["position"]


def _check_feasibility(target):
    """
    Step 2: is this object something we should even attempt?
    Placeholder for now - always feasible. This is the exact seam where
    Day 3's guardrails will plug in real size/weight/safety checks.
    """
    return True


def _check_reachability(position):
    """Step 3: can the arm's workspace actually reach this point?"""
    return check_reachable(position)


def _execute_pick(position):
    """Step 4: move to the object and close the gripper."""
    simulate_move_to(position)
    simulate_grasp()
    return True


def _verify_pick():
    """
    Step 5: confirm the pick actually worked.
    Placeholder for now - always True in simulation. Real verification
    (camera recheck or gripper feedback) plugs in here once hardware exists.
    """
    return True


def attempt_pick(target: DetectedObject) -> ToolResult:
    """
    Attempts to pick up one detected object. Chains localize -> feasibility
    -> reachability -> pick -> verify internally, so the agent makes ONE
    decision (which object) instead of five.
    """
    position = _localize(target)

    if not _check_feasibility(target):
        return {"status": "infeasible", "reason": f"{target['waste_class']} object failed feasibility check"}

    if not _check_reachability(position):
        return {"status": "unreachable", "reason": f"Position {position} is outside the arm's workspace"}

    if not _execute_pick(position):
        return {"status": "failed", "reason": "Pick execution failed"}

    if not _verify_pick():
        return {"status": "failed", "reason": "Pick could not be verified"}

    return {"status": "success", "reason": f"Picked {target['waste_class']} object at {position}"}


# ------------------------------------------------------------------


# ---------------------------------------------------------------------
# attempt_place() - Step 6
# ---------------------------------------------------------------------

def _get_target_bin_position(waste_class):
    """Step 1: look up where this waste class's bin is."""
    return get_bin_position(waste_class)


def _execute_place(position):
    """Step 2: move to the bin and open the gripper."""
    simulate_move_to(position)
    simulate_release()
    return True


def _verify_placement():
    """
    Step 3: confirm the object actually landed in the bin.
    Placeholder for now - always True in simulation. Real verification
    (a quick re-observe of the bin) plugs in here once hardware exists.
    """
    return True


def attempt_place(waste_class: str) -> ToolResult:
    """
    Attempts to place a held object into its class's bin. Chains bin
    lookup -> reachability -> move+release -> verify internally, so the
    agent makes ONE decision (nothing - this is the automatic next step
    after a successful pick) instead of three.
    """
    try:
        position = _get_target_bin_position(waste_class)
    except KeyError as e:
        return {"status": "failed", "reason": str(e).strip('"')}

    if not check_reachable(position):
        return {"status": "unreachable", "reason": f"Bin for '{waste_class}' at {position} is outside the arm's workspace"}

    if not _execute_place(position):
        return {"status": "failed", "reason": "Place execution failed"}

    if not _verify_placement():
        return {"status": "failed", "reason": "Placement could not be verified"}

    return {"status": "success", "reason": f"Placed {waste_class} object at bin {position}"}



# ---------------------------------------------------------------------
# safe_stop() - Step 7
# ---------------------------------------------------------------------

def safe_stop(reason: str) -> ToolResult:
    """
    Immediately halts the arm and ends the task. Callable from anywhere,
    any time - not just at a normal finish, but mid-pick or mid-place too
    if something looks wrong.
    """
    simulate_stop()
    print(f"[safe_stop] Task halted: {reason}")
    return {"status": "stopped", "reason": reason}
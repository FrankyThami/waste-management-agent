"""
adaptive_validation/action_tools.py

MOCKED action tools. These stand in for real robot motion for now.
Swapping to Gazebo or a real arm later means rewriting ONLY this file
- observe_scene, locate_object, and the agent loop don't change.
verify_placement is the one REAL check here - a fresh camera check,
not a blind "success" - everything else stays mocked.
"""

import numpy as np
from langchain_core.tools import tool

from perception.gemini_classifier import classify_image
from prompts.prompts_v1 import PROMPT_LOCATE_TARGETS
from adaptive_validation import scene_state
from adaptive_validation.geometry import locate_object as _locate_object_3d

MOVED_THRESHOLD_M = 0.05  # 5cm - bigger than the sensor's normal jitter, smaller than a real move


@tool
def check_pick_feasibility(target_description: str) -> str:
    """Check whether the target is within the gripper's size/weight
    limits and has clear space around it for a grasp. Call after
    locate_object(), before check_reachability()."""
    print(f"    [SIM] checking pick feasibility for '{target_description}'")
    return f"'{target_description}' is feasible to pick (mock: within limits, clear access)."


@tool
def check_reachability(x: float, y: float, z: float) -> str:
    """Check whether the robot can safely reach a 3D position (meters,
    robot frame) without collision. Call before pick_object() or
    place_in_bin()."""
    print(f"    [SIM] checking reachability of ({x}, {y}, {z})")
    return f"Position ({x}, {y}, {z}) is reachable (mock: no real workspace limits yet)."


@tool
def pick_object(target_description: str) -> str:
    """Execute the pick sequence on a target already confirmed
    feasible and reachable."""
    print(f"    [SIM] moving to '{target_description}', closing gripper")
    return f"Pick executed on '{target_description}' (mock)."


@tool
def verify_pick(target_description: str) -> str:
    """Confirm the target was actually acquired by the gripper after
    pick_object()."""
    print(f"    [SIM] checking gripper state for '{target_description}'")
    return f"'{target_description}' confirmed in gripper (mock: no real sensor feedback yet)."


@tool
def place_in_bin(bin_description: str) -> str:
    """Move the currently-held object to the destination bin and
    release it, e.g. bin_description="red dustbin"."""
    print(f"    [SIM] moving held object to '{bin_description}', opening gripper")
    return f"Placement executed at '{bin_description}' (mock)."


@tool
def verify_placement(bin_description: str) -> str:
    """Confirm the object actually landed in the intended bin after
    place_in_bin(). Does a FRESH camera check of the bin's current
    position and compares it against where it was last located - if
    the bin has moved since then, placement is reported as failed."""
    remembered_xyz = scene_state.last_known_positions.get(bin_description)
    if remembered_xyz is None:
        return f"Cannot verify - '{bin_description}' was never located before placing."

    print(f"    [SIM] re-checking real position of '{bin_description}' for verification")
    fresh_rgb = scene_state.capture_and_store()
    fresh_detections = classify_image(fresh_rgb, prompt=PROMPT_LOCATE_TARGETS)
    match = next((d for d in fresh_detections if d.get("target") == bin_description), None)

    if match is None or not match.get("found", False):
        return f"Placement verification FAILED for '{bin_description}': it is no longer visible at all."

    fresh_result = _locate_object_3d(
        match["point"], scene_state.last_color_frame, scene_state.last_depth_frame
    )
    if not fresh_result["valid"]:
        return (f"Placement verification FAILED for '{bin_description}': "
                f"could not get a fresh 3D position ({fresh_result['reason']}).")

    current_xyz = fresh_result["robot_xyz_m"]
    moved_by = float(np.linalg.norm(np.array(current_xyz) - np.array(remembered_xyz)))

    if moved_by > MOVED_THRESHOLD_M:
        return (f"Placement verification FAILED for '{bin_description}': it has moved "
                f"{moved_by*100:.1f}cm since it was last located, so the release may not "
                f"have landed inside it. Re-observe the scene to check current positions "
                f"before retrying.")

    print(f"    [SIM] confirming placement in '{bin_description}'")
    return f"Placement in '{bin_description}' confirmed (moved only {moved_by*100:.1f}cm, within tolerance)."
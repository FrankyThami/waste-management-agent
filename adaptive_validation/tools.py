"""
adaptive_validation/tools.py

Perception-side tools, wrapped for LangChain tool-calling.
"""

from langchain_core.tools import tool

from perception.gemini_classifier import classify_image
from prompts.prompts_v1 import PROMPT_LOCATE_TARGETS
from adaptive_validation import scene_state
from adaptive_validation.geometry import locate_object as _locate_object_3d


def _run_observation() -> str:
    """Shared logic behind observe_scene and reobserve_scene."""
    rgb_image = scene_state.capture_and_store()
    scene_state.last_detections = classify_image(rgb_image, prompt=PROMPT_LOCATE_TARGETS)

    lines = []
    for det in scene_state.last_detections:
        status = "visible" if det.get("found", False) else "not visible"
        lines.append(f"{det.get('target', 'unknown')}: {status}")
    return "Scene observed. " + "; ".join(lines)


@tool
def observe_scene() -> str:
    """Capture a fresh image from the camera and report which target
    objects are currently visible: "yellow paper" and "red dustbin".
    Call this first, at the start of the task."""
    return _run_observation()


@tool
def reobserve_scene() -> str:
    """Re-capture the scene when something might have changed since
    your last observation - after a failure, an unexpected result from
    another tool, or any uncertainty about where things currently are.
    Positions you located before this may now be stale; re-locate
    anything you need after calling this."""
    return _run_observation()


@tool
def locate_object(target_description: str) -> str:
    """Get the precise 3D position of a target that a scene
    observation already confirmed is visible. target_description must
    be exactly "yellow paper" or "red dustbin". Uses the most recent
    observation - call observe_scene() or reobserve_scene() again
    first if the scene may have changed."""
    if scene_state.last_detections is None:
        return "Error: observe the scene before calling locate_object()."

    match = next(
        (d for d in scene_state.last_detections if d.get("target") == target_description),
        None,
    )
    if match is None or not match.get("found", False):
        return f"'{target_description}' is not visible in the current scene."

    result = _locate_object_3d(
        match["point"], scene_state.last_color_frame, scene_state.last_depth_frame
    )
    if not result["valid"]:
        return f"Found '{target_description}' in 2D but 3D lookup failed: {result['reason']}"

    scene_state.last_known_positions[target_description] = result["robot_xyz_m"]

    x, y, z = result["robot_xyz_m"]
    return f"'{target_description}' is at approximately x={x}, y={y}, z={z} meters (robot frame)."
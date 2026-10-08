"""
observation.py - one full look at the table, for the agent's observe_scene tool (Stage 2.5a).

Camera (depth combined over several frames) -> Gemini finds objects and bins ->
sanity rules -> 3D positions in the robot-base frame -> plain dictionaries for the agent.
Nothing is decided here: it only reports what is on the table, and (Stage 3.1) what
changed since the previous look.
"""

import json
import os
import time
from datetime import datetime

import cv2

from src.actions.camera import capture_frame
from src.actions.localization import localize_scene
from src.actions.scene_mirror import mirror_bins
from src.context.scene_diff import compare_observations
from src.interface.scene_view import draw_scene
from src.models.gemini_classifier import ClassificationError
from src.models.scene_detector import build_scene
from src.reliability import run_stats
from src.reliability.scene_checks import clean_scene

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBSERVATIONS_DIR = os.path.join(PROJECT_ROOT, "logs", "observations")

# Saving every look as images + JSON is OFF while we test (it fills the logs folder).
# To turn it on for one run, start PowerShell and type first:  $env:SAVE_OBSERVATIONS = "1"
SAVE_OBSERVATIONS = os.environ.get("SAVE_OBSERVATIONS") == "1"

# The previous successful look (objects + bins) and when it was taken. Lives as long as the program runs.
_last_look = None


def reset_look_memory():
    """Forget the previous look, so the first look of a NEW task is not compared with the last task's scene."""
    global _last_look
    _last_look = None


def _position_dict(point):
    x, y, z = point
    return {"x": x, "y": y, "z": z}


def _box_size_m(box_px, depth_m, intrinsics):
    """
    Rough real-world size (width, height) of a pixel box, in metres.
    Pinhole camera maths: real size = size in pixels x distance / focal length.
    Gemini's boxes are a little loose, so this slightly over-estimates.
    """
    x1, y1, x2, y2 = box_px
    return round((x2 - x1) * depth_m / intrinsics.fx, 3), round((y2 - y1) * depth_m / intrinsics.fy, 3)


def _save_observation(frame, scene):
    """Saves the raw frame, the annotated image and the Scene as JSON, so every observation can be reviewed later."""
    try:
        os.makedirs(OBSERVATIONS_DIR, exist_ok=True)
        base = os.path.join(OBSERVATIONS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        scene.frame_path = f"{base}_raw.png"
        cv2.imwrite(scene.frame_path, frame.color_bgr)
        cv2.imwrite(f"{base}_annotated.png", draw_scene(frame.color_bgr, scene))
        with open(f"{base}_scene.json", "w") as f:
            json.dump(scene.to_dict(), f, indent=2)
        print(f"[observe_scene] Saved this observation: {base}_annotated.png (+ _raw.png, _scene.json)")
    except OSError as error:
        print(f"[observe_scene] Could not save the observation ({error}) - carrying on")


def _compare_with_last_look(observation, remember):
    """
    Compares this look with the previous one and returns the change report
    (None on the very first look, when there is nothing to compare with).
    If remember is True, this look also becomes the "previous look" for next time.
    Re-check looks (remember=False) leave it alone, so the baseline stays what the agent was last shown.
    """
    global _last_look
    now = time.time()
    changes = None
    if _last_look is not None:
        changes = compare_observations(
            _last_look["observation"], observation, round(now - _last_look["time"], 1)
        )
        if changes["anything_changed"]:
            print("[observe_scene] The scene has changed since the previous look.")
    if remember:
        _last_look = {
            "observation": {"objects": observation["objects"], "bins": observation["bins"]},
            "time": now,
        }
    return changes


def observe(remember: bool = True) -> dict:   # remember=False: a re-check look, not the new "previous look"
    """
    Looks at the table once and returns:
      {"objects":    [{"id", "waste_class", "position", "confidence", "width_m", "height_m"}, ...],
       "bins":       [{"id", "colour", "position", "width_m", "height_m"}, ...],
       "unmeasured": ["obj_3 (plastic)", ...],   seen, but no reliable depth - never guessed
       "changes":    what changed since the previous look (None on the first look),
       "error":      None, or text if the vision step failed}
    Positions are metres in the robot-base frame (z = height above the tabletop;
    a bin's position is its rim, where an item is released).
    Ids (obj_0, bin_1, ...) are only valid inside ONE look; "changes" says which old id is which new id.
    """
    frame = capture_frame()

    run_stats.count_vision_look()   # every look is one Gemini request; counted for the run summary
    try:
        scene = build_scene(frame.to_pil())
    except ClassificationError as error:
        print(f"[observe_scene] Gemini call failed: {error}")
        return {"objects": [], "bins": [], "unmeasured": [], "changes": None, "error": str(error)}

    clean_scene(scene)
    localize_scene(scene, frame)
    print(f"[observe_scene] Looked at the table: {len(scene.objects)} object(s), {len(scene.bins)} bin(s).")
    if SAVE_OBSERVATIONS:
        _save_observation(frame, scene)

    objects, bins, unmeasured = [], [], []

    for obj in scene.objects:
        if obj.position_base is None:
            unmeasured.append(f"{obj.object_id} ({obj.waste_class})")
            continue
        width_m, height_m = _box_size_m(obj.box_px, obj.position_camera[2], frame.intrinsics)
        objects.append({
            "id": obj.object_id,
            "waste_class": obj.waste_class,
            "position": _position_dict(obj.position_base),
            "confidence": 1.0,   # Gemini is not asked for a confidence yet - placeholder, as before
            "width_m": width_m,
            "height_m": height_m,
        })

    for b in scene.bins:
        if b.position_base is None:
            unmeasured.append(f"{b.bin_id} ({b.colour} bin)")
            continue
        width_m, height_m = _box_size_m(b.box_px, b.position_camera[2], frame.intrinsics)
        bins.append({
            "id": b.bin_id,
            "colour": b.colour,
            "position": _position_dict(b.position_base),
            "width_m": width_m,
            "height_m": height_m,
        })

    mirror_bins(bins)   # draw the real bins as coloured boxes in the MATLAB window (display only)

    observation = {"objects": objects, "bins": bins, "unmeasured": unmeasured, "error": None}
    observation["changes"] = _compare_with_last_look(observation, remember)
    return observation
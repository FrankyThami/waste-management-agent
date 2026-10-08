"""
scripts/check_scene_vision.py - live check of what the agent sees.

Capture (depth combined over several frames) -> Gemini finds objects and bins ->
sanity rules clean the scene -> 3D positions in robot-base coordinates -> annotated image.

Normal run:          python scripts/check_scene_vision.py
With stability test: python scripts/check_scene_vision.py --stability
   (measures every item 10 more times to show whether depth is found reliably)
"""

import copy
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# Let this script import from src/ no matter where it is started from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.actions.camera import capture_frame              # noqa: E402
from src.actions.localization import localize_scene       # noqa: E402
from src.models.scene_detector import build_scene         # noqa: E402
from src.reliability.scene_checks import clean_scene      # noqa: E402
from src.interface.scene_view import draw_scene           # noqa: E402

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logs", "scene_checks")
STABILITY_REPEATS = 10


def _items(scene):
    """(id, label, item) for every object and bin in the scene."""
    rows = [(o.object_id, o.waste_class, o) for o in scene.objects]
    rows += [(b.bin_id, f"{b.colour} bin rim", b) for b in scene.bins]
    return rows


def print_positions(scene):
    """One row per object and bin: pixel centre and robot-base position in metres."""
    print("Robot-base frame, metres: x = away from the robot (toward the image bottom), "
          "y = toward the image right, z = height above the tabletop")
    for item_id, label, item in _items(scene):
        px, py = item.center_px
        if item.position_base is None:
            print(f"  {item_id:<6} {label:<16} pixel=({px:>4},{py:>4})  not measured (no reliable depth)")
        else:
            x, y, z = item.position_base
            print(f"  {item_id:<6} {label:<16} pixel=({px:>4},{py:>4})  x={x:+.3f}  y={y:+.3f}  z={z:+.3f}")


def check_depth_stability(scene):
    """
    Measures every item again with the real camera (same boxes, brand-new frames) and
    reports how often depth was found and how much the height varied.
    """
    print(f"\nDepth stability: measuring {STABILITY_REPEATS} more times with the same boxes...")
    items = _items(scene)
    heights = {item_id: [] for item_id, _, _ in items}

    for _ in range(STABILITY_REPEATS):
        trial = copy.deepcopy(scene)
        localize_scene(trial, capture_frame())
        for (item_id, _, _), (_, _, trial_item) in zip(items, _items(trial)):
            if trial_item.position_base is not None:
                heights[item_id].append(trial_item.position_base[2])

    print("Per item: how many of the measurements found depth, and the height spread:")
    for item_id, label, _ in items:
        values = heights[item_id]
        if not values:
            print(f"  {item_id:<6} {label:<16} depth found 0/{STABILITY_REPEATS}")
            continue
        spread_mm = (max(values) - min(values)) * 1000
        print(f"  {item_id:<6} {label:<16} depth found {len(values)}/{STABILITY_REPEATS}   "
              f"z = {np.mean(values) * 100:4.1f} cm   spread (max - min) = {spread_mm:.1f} mm")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Capturing frames from the RealSense D435i...")
    frame = capture_frame()

    raw_path = os.path.join(OUTPUT_DIR, f"{stamp}_raw.png")
    cv2.imwrite(raw_path, frame.color_bgr)

    print("Asking Gemini Robotics ER 2 what it sees...")
    started = time.perf_counter()
    scene = build_scene(frame.to_pil(), frame_path=raw_path)
    elapsed = time.perf_counter() - started

    clean_scene(scene)
    localize_scene(scene, frame)

    print(f"\nScene at {scene.timestamp}: {len(scene.objects)} object(s), {len(scene.bins)} bin(s)")
    print_positions(scene)
    print(f"\nGemini answered in {elapsed:.1f} s")

    annotated_path = os.path.join(OUTPUT_DIR, f"{stamp}_annotated.png")
    cv2.imwrite(annotated_path, draw_scene(frame.color_bgr, scene))
    print(f"Annotated image saved to: {annotated_path}")

    try:
        os.startfile(annotated_path)  # Windows: opens in your default image viewer
    except (AttributeError, OSError):
        print("(Open the image above manually to check it.)")

    if "--stability" in sys.argv:
        check_depth_stability(scene)


if __name__ == "__main__":
    main()
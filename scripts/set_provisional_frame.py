"""
scripts/set_provisional_frame.py - Stage 2.3b (provisional frame).

Builds a PROVISIONAL camera -> robot-base transform from the table plane plus one
rough distance, saves it to config/camera_to_base_transform.json, then shows what
the agent now sees in robot-base coordinates.

Keep your normal scene on the table (bins and items). No tape or measuring needed.

Run from the project root:   python scripts/set_provisional_frame.py
"""

import os
import sys

import numpy as np

# Let this script import from src/ no matter where it is started from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.actions.camera import capture_frame                         # noqa: E402
from src.actions.localization import localize_scene                  # noqa: E402
from src.actions.table_plane import fit_table_plane                  # noqa: E402
from src.actions.frame_transform import (                            # noqa: E402
    build_provisional_transform, save_transform, table_point_at_pixel, TRANSFORM_PATH,
)
from src.models.scene_detector import build_scene                    # noqa: E402


def ask_robot_distance_m():
    print("How far is the CENTRE of the robot base from the TOP EDGE of the camera image?")
    print("Measure along the table, in cm. A rough number is fine - it can be changed later.")
    raw = input("Distance (cm): ").strip().replace(",", ".")
    try:
        value_cm = float(raw)
    except ValueError:
        print("Please type a number.")
        return None
    if not 0 <= value_cm <= 100:
        print("Please type a number between 0 and 100 cm.")
        return None
    return value_cm / 100.0


def main():
    distance_m = ask_robot_distance_m()
    if distance_m is None:
        return

    print("\nCapturing a frame from the RealSense D435i...")
    frame = capture_frame()

    print("Asking Gemini where the objects and bins are (so they are left out of the table fit)...")
    scene = build_scene(frame.to_pil())
    boxes = [item.box_px for item in [*scene.objects, *scene.bins]]

    plane = fit_table_plane(frame.depth_m, frame.intrinsics, exclude_boxes=boxes)
    if plane is None:
        print("Could not fit the table plane - not enough bare table in view.")
        return
    print(f"Table plane: camera height {plane.camera_height_m * 100:.1f} cm, "
          f"tilt {plane.tilt_deg:.2f} deg, flatness {plane.rms_m * 1000:.1f} mm")

    result = build_provisional_transform(plane, frame.intrinsics, frame.width, distance_m)
    rotation = np.array(result["rotation_matrix"])
    translation = np.array(result["translation_vector_m"])

    print("\nRobot frame (PROVISIONAL): z = up from the tabletop, "
          "x = toward the image bottom, y = toward the image right")
    print(f"  camera position in the robot frame: "
          f"x={translation[0]:+.3f}  y={translation[1]:+.3f}  z={translation[2]:+.3f} m")

    corners = [table_point_at_pixel(plane, frame.intrinsics, px, py)
               for px, py in [(0, 0), (frame.width - 1, 0), (0, frame.height - 1), (frame.width - 1, frame.height - 1)]]
    corners_base = np.array([rotation @ c + translation for c in corners])
    print(f"  the camera view covers: x from {corners_base[:, 0].min():.2f} to {corners_base[:, 0].max():.2f} m, "
          f"y from {corners_base[:, 1].min():+.2f} to {corners_base[:, 1].max():+.2f} m")

    answer = input("\nSave this PROVISIONAL frame to config/camera_to_base_transform.json? (y/n): ").strip().lower()
    if answer != "y":
        print("Not saved.")
        return

    status = (f"PROVISIONAL - table plane + one rough distance (robot base assumed {distance_m * 100:.0f} cm "
              "beyond the top edge of the image, on the image's centre line). "
              "Replace with a real calibration before judging real-arm accuracy.")
    save_transform(result, status, "Built by scripts/set_provisional_frame.py")
    print(f"Saved to {TRANSFORM_PATH}")

    localize_scene(scene, frame)
    print("\nWhat the agent now sees, in robot-base coordinates (metres, z = height above the tabletop):")
    for item_id, label, position in (
        [(o.object_id, o.waste_class, o.position_base) for o in scene.objects]
        + [(b.bin_id, f"{b.colour} bin rim", b.position_base) for b in scene.bins]
    ):
        if position is None:
            print(f"  {item_id:<6} {label:<16} not measured")
        else:
            print(f"  {item_id:<6} {label:<16} x={position[0]:+.3f}  y={position[1]:+.3f}  z={position[2]:+.3f}")


if __name__ == "__main__":
    main()
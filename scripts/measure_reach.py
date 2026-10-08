"""
scripts/measure_reach.py - finds where the robot base must stand so the arm can reach the whole workspace.

Uses the REAL MATLAB simulation (matlab_server must be running). For each possible distance between
the robot base and the table's top edge, it asks the simulation whether the fingertips can reach:
  - every pick point and bin release point in the last saved observation (logs/observations/*_scene.json)
  - nine spots spread over your whole tape rectangle (corners, edge middles, centre), so an item
    placed ANYWHERE on the table can be reached, not only where it lies today
  - the travel height above each of those points
Nothing moves - it only checks.

Before running:
  1. Put your normal scene on the table: ALL your bins, items OUTSIDE the bins.
  2. python scripts/check_observe_tool.py        (saves a fresh observation)
  3. Keep matlab_server running in MATLAB.

Run from the project root:   python scripts/measure_reach.py
"""

import glob
import json
import os
import sys

# Let this script import from src/ no matter where it is started from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import policy                                   # noqa: E402
from src.actions import matlab_hal                       # noqa: E402
from src.actions.frame_transform import TRANSFORM_PATH   # noqa: E402

CANDIDATES_M = [round(0.10 + 0.02 * i, 2) for i in range(21)]   # 10 cm ... 50 cm from the table's top edge

# Your masking-tape rectangle (measured on the table). The tape's top edge sits about 3 cm
# below the top edge of the camera image. If you change the tape, change these numbers.
TABLE_WIDTH_M = 0.51
TABLE_DEPTH_M = 0.27
TABLE_TOP_OFFSET_M = 0.03
PICK_HEIGHT_M = 0.04    # about the top of the tallest item you use


def newest_scene():
    paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, "logs", "observations", "*_scene.json")))
    if not paths:
        return None, None
    with open(paths[-1]) as f:
        return paths[-1], json.load(f)


def build_points(scene, current_m):
    """Every point the arm must reach: (label, x, y, z) in the CURRENT robot frame, metres."""
    points = []
    hover = matlab_hal.SAFE_TRAVEL_Z   # the height the arm travels at between places

    for obj in scene["objects"]:
        if obj.get("position_base"):
            x, y, z = obj["position_base"]
            label = f"{obj['object_id']} ({obj['waste_class']})"
            points.append((f"{label} pick", x, y, z))
            points.append((f"{label} travel height", x, y, max(hover, z)))

    for b in scene["bins"]:
        if b.get("position_base"):
            x, y, z = b["position_base"]
            z = z + policy.RELEASE_CLEARANCE_M    # fingertips release above the rim
            label = f"{b['colour']} bin"
            points.append((f"{label} release", x, y, z))
            points.append((f"{label} travel height", x, y, max(hover, z)))

    # Nine spots over the whole table: image-top / middle / image-bottom  x  image-left / centre / image-right.
    top = current_m + TABLE_TOP_OFFSET_M
    rows = [("top edge", top), ("middle", top + TABLE_DEPTH_M / 2), ("bottom edge", top + TABLE_DEPTH_M)]
    columns = [("image-left", -TABLE_WIDTH_M / 2), ("centre", 0.0), ("image-right", TABLE_WIDTH_M / 2)]
    for row_name, x in rows:
        for column_name, y in columns:
            label = f"table {row_name}, {column_name}"
            points.append((f"{label} pick", x, y, PICK_HEIGHT_M))
            points.append((f"{label} travel height", x, y, hover))
    return points


def longest_run(distances):
    """The longest stretch of consecutive candidate distances in a set."""
    runs, run = [], []
    for d in CANDIDATES_M:
        if d in distances:
            run.append(d)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return max(runs, key=len) if runs else []


def main():
    path, scene = newest_scene()
    if scene is None:
        print("No saved observation found. Run:  python scripts/check_observe_tool.py")
        return

    with open(TRANSFORM_PATH) as f:
        current_m = json.load(f).get("robot_distance_m")
    if current_m is None:
        print("config/camera_to_base_transform.json has no robot_distance_m. "
              "Run:  python scripts/set_provisional_frame.py")
        return

    points = build_points(scene, current_m)

    try:
        client = matlab_hal._get_client()
    except matlab_hal.MatlabBridgeError as error:
        print(f"Could not connect to the MATLAB simulation: {error}")
        print("Is matlab_server running in MATLAB?")
        return

    bins_seen = ", ".join(b["colour"] for b in scene["bins"]) or "none"
    print(f"Using {os.path.basename(path)}: {len(scene['objects'])} object(s); bins in this observation: {bins_seen}.")
    print(f"{len(points)} points to reach: items and bins as they lie today, nine spots over the whole table, "
          "and the travel height above each.")
    print(f"Positions were measured with the robot base {current_m * 100:.0f} cm from the table's top edge.\n")

    reachable_at = set()
    fewest = None   # (number failing, distance, labels, first reason)

    for d in CANDIDATES_M:
        shift = d - current_m    # moving the base further away shifts every x by the same amount
        failing, first_reason = [], None
        for label, x, y, z in points:
            try:
                reply = client.check_position(x + shift, y, z)
            except matlab_hal.MatlabBridgeError as error:
                print(f"Lost the connection to MATLAB: {error}")
                return
            if reply.get("status") != "ok":
                failing.append(label)
                first_reason = first_reason or reply.get("reason")

        if not failing:
            reachable_at.add(d)
            print(f"  base {d * 100:3.0f} cm from the table's top edge: all {len(points)} points reachable")
        else:
            shown = ", ".join(failing[:3]) + (" ..." if len(failing) > 3 else "")
            print(f"  base {d * 100:3.0f} cm from the table's top edge: "
                  f"{len(failing)} of {len(points)} NOT reachable ({shown})")
            if fewest is None or len(failing) < fewest[0]:
                fewest = (len(failing), d, failing, first_reason)

    print()
    window = longest_run(reachable_at)
    if window:
        low, high = window[0], window[-1]
        recommended = round((low + high) / 2, 2)
        print(f"The whole workspace is reachable when the base is between {low * 100:.0f} and {high * 100:.0f} cm "
              "from the table's top edge.")
        print(f"Recommended: {recommended * 100:.0f} cm (the middle of that window, so there is margin on both sides).")
        if len(window) == 1:
            print("The window is only one step wide - the table is close to the limit of what the arm can cover.")
    else:
        count, d, labels, reason = fewest
        print("No distance made every point reachable.")
        print(f"Closest: base {d * 100:.0f} cm, with {count} point(s) failing: {', '.join(labels)}")
        print(f"Simulation said: {reason}")


if __name__ == "__main__":
    main()
"""
src/actions/matlab_hal.py

The MATLAB backend for the HAL. hal.py uses this ONLY when HAL_MODE=matlab.
Each function sends commands to the MATLAB simulation (matlab_server.m)
and returns True/False, just like the terminal-simulation functions.

All positions are FINGERTIP positions in metres (robot base frame, z up, ground = 0).

Safety rule: if anything goes wrong (MATLAB not running, timeout, error),
we print the reason and return False. We never pretend a move worked.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

# matlab_bridge.py lives in the matlabrobot folder, so tell Python to look there too.
MATLAB_DIR = os.path.join(PROJECT_ROOT, "matlabrobot")
if MATLAB_DIR not in sys.path:
    sys.path.append(MATLAB_DIR)

from matlab_bridge import MatlabRobotClient, MatlabBridgeError  # noqa: E402

# Fingertip height used when travelling sideways between places (metres).
# EXAMPLE value: it must be higher than the tallest object or bin wall plus some clearance.
# Override for one terminal with:   $env:HAL_TRAVEL_Z = "0.25"
SAFE_TRAVEL_Z = float(os.environ.get("HAL_TRAVEL_Z", "0.20"))

_client = None            # the open connection to MATLAB (None = not connected)
_arm_was_reset = False    # has this Python run already reset the simulated arm?


def _get_client():
    """Connect on first use, then reuse the same connection."""
    global _client, _arm_was_reset
    if _client is None:
        client = MatlabRobotClient()
        client.connect()              # raises MatlabBridgeError if MATLAB is not listening
        if not _arm_was_reset:
            client.reset()            # every new run starts from the home pose
            _arm_was_reset = True
        _client = client
    return _client


def _ask(action, description):
    """
    Run one command on the MATLAB client and return MATLAB's reply (a dict).
    Returns None if MATLAB could not be reached.
    'action' is a tiny function that receives the client and sends one command.
    """
    global _client
    try:
        return action(_get_client())
    except MatlabBridgeError as error:
        _client = None                # forces a fresh connection next time
        print(f"[HAL-MATLAB] ERROR while trying to {description}: {error}")
        return None


def check_reachable(position):
    """Ask MATLAB if the fingertips can go to this {x, y, z}. Does not move the arm."""
    reply = _ask(lambda c: c.check_position(position["x"], position["y"], position["z"]),
                 "check reachability")
    if reply is None:
        return False
    if reply.get("status") == "ok":
        return True
    print(f"[HAL-MATLAB] Not reachable: {reply.get('reason')}")
    return False


def _plan_waypoints(tip, target):
    """
    Turn 'go from here to the target' into: lift, travel sideways, descend.
    tip    = current fingertip position [x, y, z]
    target = {'x', 'y', 'z'}
    Returns a list of (x, y, z) points. Points we are already at are dropped.
    If we are already at the target, the list is empty: no motion at all.
    """
    already_there = (abs(tip[0] - target["x"]) <= 0.001
                     and abs(tip[1] - target["y"]) <= 0.001
                     and abs(tip[2] - target["z"]) <= 0.001)
    if already_there:
        return []

    hover_z = max(SAFE_TRAVEL_Z, tip[2], target["z"])
    points = [
        (tip[0], tip[1], hover_z),                      # 1. straight up
        (target["x"], target["y"], hover_z),            # 2. sideways at the travel height
        (target["x"], target["y"], target["z"]),        # 3. straight down to the target
    ]
    kept = []
    last = tuple(tip)
    for point in points:
        if max(abs(a - b) for a, b in zip(point, last)) > 0.001:
            kept.append(point)
            last = point
    return kept


def move_to(position, animate=True):
    """
    Move the fingertips to this {x, y, z}: lift, travel at the safe height, descend.
    All waypoints are checked BEFORE anything moves. True only if every segment finished.
    """
    print(f"[HAL-MATLAB] Moving arm to {position}")

    state = _ask(lambda c: c.get_state(), "read the arm position")
    if state is None or "tip" not in state:
        print("[HAL-MATLAB] Could not read the arm position - not moving")
        return False

    waypoints = _plan_waypoints(state["tip"], position)

    # 1. Check every waypoint first. If one is unreachable, the arm does not move at all.
    for x, y, z in waypoints:
        reply = _ask(lambda c, x=x, y=y, z=z: c.check_position(x, y, z), "check a waypoint")
        if reply is None:
            return False
        if reply.get("status") != "ok":
            print(f"[HAL-MATLAB] Move refused before moving: waypoint "
                  f"({x:.3f}, {y:.3f}, {z:.3f}) - {reply.get('reason')}")
            return False

    # 2. Move through the waypoints, one segment at a time.
    for x, y, z in waypoints:
        reply = _ask(lambda c, x=x, y=y, z=z: c.move_to(x, y, z, animate), "move the arm")
        if reply is None:
            return False
        if reply.get("status") != "done":
            print(f"[HAL-MATLAB] Move refused: {reply.get('status')} - {reply.get('reason')}")
            return False
    return True


def go_home(animate=True):
    """
    Send the fingertips to the home pose (read from MATLAB), using the same
    lift -> travel -> descend moves. True only if the arm got there.
    """
    state = _ask(lambda c: c.get_state(), "read the home position")
    if state is None or "home" not in state:
        print("[HAL-MATLAB] Could not read the home position - not moving")
        return False
    x, y, z = state["home"]
    print("[HAL-MATLAB] Going to the home pose")
    return move_to({"x": x, "y": y, "z": z}, animate=animate)


def set_gripper(state):
    """state is 'open' or 'closed'. True only if MATLAB confirmed it."""
    reply = _ask(lambda c: c.set_gripper(state), f"set the gripper to {state}")
    if reply is None:
        return False
    if reply.get("status") == "done":
        print(f"[HAL-MATLAB] Gripper -> {state}")
        return True
    print(f"[HAL-MATLAB] Gripper command refused: {reply.get('reason')}")
    return False


def stop():
    """Halt the simulated arm. It refuses moves until reset() is called."""
    reply = _ask(lambda c: c.stop(), "stop the arm")
    return reply is not None and reply.get("status") == "done"


def reset():
    """Back to the home pose, gripper open, stop cleared."""
    reply = _ask(lambda c: c.reset(), "reset the arm")
    return reply is not None and reply.get("status") == "done"


def get_bin_position(waste_class):
    """Bin release positions for MATLAB mode (fingertip positions)."""
    path = os.path.join(CONFIG_DIR, "matlab_bin_positions.json")
    with open(path, "r") as f:
        data = json.load(f)
    bins = data["bins"]
    if waste_class not in bins:
        raise KeyError(f"No bin position configured for class '{waste_class}'")
    x, y, z = bins[waste_class]
    return {"x": x, "y": y, "z": z}


def show_boxes(boxes):
    """Draw the table boxes in the MATLAB window (visual only). True only if MATLAB confirmed it."""
    reply = _ask(lambda c: c.set_boxes(boxes), "draw the table boxes")
    if reply is None:
        return False
    if reply.get("status") == "done":
        return True
    print(f"[HAL-MATLAB] Box drawing refused: {reply.get('reason')}")
    return False
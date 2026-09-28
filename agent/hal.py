"""
Hardware Abstraction Layer (HAL)

This stands in for the real robot arm. Every function here simulates a
physical action and always succeeds in a predictable way, so the rest of
the agent can be built and tested before the arm exists.

When the real arm arrives, only THIS file changes - tools.py and the
agent loop never need to know the difference.
"""

import json
import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def _load_json(filename):
    path = os.path.join(CONFIG_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)


# Placeholder reachable envelope in the robot base frame (metres).
# EXAMPLE VALUES - replace with your arm's real reach once you know it.
WORKSPACE_BOUNDS = {
    "x": (-0.4, 0.4),
    "y": (-0.4, 0.4),
    "z": (0.0, 0.5),
}


def check_reachable(position):
    """
    position: dict with 'x', 'y', 'z' in the robot base frame (metres).
    Returns True if the position falls inside the simulated workspace.
    """
    x_ok = WORKSPACE_BOUNDS["x"][0] <= position["x"] <= WORKSPACE_BOUNDS["x"][1]
    y_ok = WORKSPACE_BOUNDS["y"][0] <= position["y"] <= WORKSPACE_BOUNDS["y"][1]
    z_ok = WORKSPACE_BOUNDS["z"][0] <= position["z"] <= WORKSPACE_BOUNDS["z"][1]
    return x_ok and y_ok and z_ok


def simulate_move_to(position):
    """Pretend to move the arm to a position. Always succeeds in simulation."""
    print(f"[HAL-SIM] Moving arm to {position}")
    return True


def simulate_grasp():
    """Pretend to close the gripper. Always succeeds in simulation."""
    print("[HAL-SIM] Closing gripper")
    return True


def simulate_release():
    """Pretend to open the gripper. Always succeeds in simulation."""
    print("[HAL-SIM] Opening gripper")
    return True

def simulate_stop():
    """Pretend to immediately halt all arm motion. Always succeeds in simulation."""
    print("[HAL-SIM] STOP - halting all motion")
    return True


def get_bin_position(waste_class):
    """Look up the target bin position for a waste class, as an {x, y, z} dict."""
    data = _load_json("bin_positions.json")
    bins = data["bins"]
    if waste_class not in bins:
        raise KeyError(f"No bin position configured for class '{waste_class}'")
    x, y, z = bins[waste_class]
    return {"x": x, "y": y, "z": z}

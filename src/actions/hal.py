"""
Hardware Abstraction Layer (HAL)

This stands in for the real robot arm. Every function here simulates a
physical action, so the rest of the agent can be built and tested before
the arm exists. By default everything succeeds; failures can be switched
on (see agent/policy.py, or inject_failures below) to test recovery.

When the real arm arrives, only THIS file changes - tools.py and the
agent loop never need to know the difference.

Two backends, chosen with the HAL_MODE environment variable:
  terminal (default) - the original print-only simulation
  matlab             - motion is simulated in MATLAB (see matlab_hal.py)
"""

import json
import os
import random

from src import policy

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config")


# --- NEW: choose which backend to use ---------------------------------
# In PowerShell, before running the agent:   $env:HAL_MODE = "matlab"
# If HAL_MODE is not set, the original terminal simulation is used.
HAL_MODE = os.environ.get("HAL_MODE", "terminal").strip().lower()
if HAL_MODE not in ("terminal", "matlab"):
    raise ValueError(f"HAL_MODE must be 'terminal' or 'matlab', but it is '{HAL_MODE}'")

# In MATLAB mode: "1" = show the arm moving (slower), "0" = just compute (faster).
MATLAB_ANIMATE = os.environ.get("HAL_ANIMATE", "1") == "1"

# "1" = park the arm at the home pose before every observe_scene (so it is out of the camera's view).
HOME_ON_OBSERVE = os.environ.get("HAL_HOME_ON_OBSERVE", "1") == "1"
if HAL_MODE == "matlab":
    from src.actions import matlab_hal
# --- end of NEW -------------------------------------------------------


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

# --- Simulated failures (testing only) ---
_forced_failures = {
    "grasp": policy.SIM_FORCED_GRASP_FAILURES,
    "release": policy.SIM_FORCED_RELEASE_FAILURES,
}
_failure_rates = {
    "grasp": policy.SIM_GRASP_FAILURE_RATE,
    "release": policy.SIM_RELEASE_FAILURE_RATE,
}


def inject_failures(grasp=0, release=0):
    """Make the next `grasp` grasps and the next `release` releases fail."""
    _forced_failures["grasp"] = grasp
    _forced_failures["release"] = release


def set_failure_rates(grasp=0.0, release=0.0):
    """Make every grasp / release fail at random with this probability (0.0 to 1.0)."""
    _failure_rates["grasp"] = grasp
    _failure_rates["release"] = release


def _should_fail(kind):
    if _forced_failures[kind] > 0:
        _forced_failures[kind] -= 1
        return True
    return random.random() < _failure_rates[kind]


def check_reachable(position):
    """
    position: dict with 'x', 'y', 'z' in the robot base frame (metres).
    Returns True if the position falls inside the simulated workspace.
    In MATLAB mode, MATLAB decides (safe box + joint limits + inverse kinematics).
    """
    if HAL_MODE == "matlab":  # NEW
        return matlab_hal.check_reachable(position)

    x_ok = WORKSPACE_BOUNDS["x"][0] <= position["x"] <= WORKSPACE_BOUNDS["x"][1]
    y_ok = WORKSPACE_BOUNDS["y"][0] <= position["y"] <= WORKSPACE_BOUNDS["y"][1]
    z_ok = WORKSPACE_BOUNDS["z"][0] <= position["z"] <= WORKSPACE_BOUNDS["z"][1]
    return x_ok and y_ok and z_ok


def simulate_move_to(position):
    """Pretend to move the arm to a position. Always succeeds in terminal simulation."""
    if HAL_MODE == "matlab":  # NEW
        return matlab_hal.move_to(position, animate=MATLAB_ANIMATE)

    print(f"[HAL-SIM] Moving arm to {position}")
    return True


def simulate_grasp():
    """Pretend to close the gripper. Returns False if a failure was simulated."""
    if _should_fail("grasp"):
        print("[HAL-SIM] Closing gripper - FAILED (simulated)")
        return False
    if HAL_MODE == "matlab":  # NEW
        return matlab_hal.set_gripper("closed")

    print("[HAL-SIM] Closing gripper")
    return True


def simulate_release():
    """Pretend to open the gripper. Returns False if a failure was simulated."""
    if _should_fail("release"):
        print("[HAL-SIM] Opening gripper - FAILED (simulated)")
        return False
    if HAL_MODE == "matlab":  # NEW
        return matlab_hal.set_gripper("open")

    print("[HAL-SIM] Opening gripper")
    return True


def simulate_stop():
    """Pretend to immediately halt all arm motion. Always succeeds in terminal simulation."""
    if HAL_MODE == "matlab":  # NEW
        print("[HAL-MATLAB] STOP - halting all motion")
        return matlab_hal.stop()

    print("[HAL-SIM] STOP - halting all motion")
    return True


def reset_simulation():  # NEW
    """Return the simulated arm to its start state. Does nothing in terminal mode."""
    if HAL_MODE == "matlab":
        return matlab_hal.reset()
    return True


def show_scene_boxes(boxes):
    """Draw the real table's boxes in the MATLAB window (visual only). Does nothing in terminal mode."""
    if HAL_MODE == "matlab":
        return matlab_hal.show_boxes(boxes)
    return True


def go_home():
    """Send the arm to the home pose (gripper down, fingertips low, parked at the side)."""
    if HAL_MODE == "matlab":
        return matlab_hal.go_home(animate=MATLAB_ANIMATE)
    print("[HAL-SIM] Moving arm to the home pose")
    return True


def go_home_for_observation():
    """
    Called at the start of observe_scene(): parks the arm out of the camera's view.
    If the arm cannot get home, we warn and carry on: looking never moves anything.
    """
    if not HOME_ON_OBSERVE:
        return True
    ok = go_home()
    if not ok:
        print("[HAL] WARNING: the arm could not reach the home pose - observing anyway")
    return ok


def get_bin_position(waste_class):
    """Look up the target bin position for a waste class, as an {x, y, z} dict."""
    if HAL_MODE == "matlab":  # NEW
        return matlab_hal.get_bin_position(waste_class)

    data = _load_json("bin_positions.json")
    bins = data["bins"]
    if waste_class not in bins:
        raise KeyError(f"No bin position configured for class '{waste_class}'")
    x, y, z = bins[waste_class]
    return {"x": x, "y": y, "z": z}
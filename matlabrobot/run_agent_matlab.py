"""
run_agent_matlab.py - real agent (updated for the downward gripper, G4b)
Runs your REAL agent (src.interface.run_agent) with:
  - the HAL in MATLAB mode (the arm moves in MATLAB)
  - a SCRIPTED scene (from sim_layout.py) instead of the camera + Gemini vision
You still type the command, exactly as you do today.
MATLAB must already be running matlab_server.

Run from the project root, for example:
  python matlabrobot\\run_agent_matlab.py "paper into the red bin"
  python matlabrobot\\run_agent_matlab.py --sort-all
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(1, HERE)

os.environ["HAL_MODE"] = "matlab"            # must be set BEFORE the agent code is imported
os.environ.setdefault("HAL_ANIMATE", "1")    # "1" = show the arm moving

from src.actions import hal, langchain_tools  # noqa: E402
import sim_layout  # noqa: E402

REMAINING = list(sim_layout.OBJECTS)         # a copy, so the layout file itself never changes
held = {"item": None}                        # the object the gripper is currently holding


def fake_observe_scene():
    """Replaces the camera + Gemini vision. Returns what is still 'on the table'."""
    hal.go_home_for_observation()    # the real observe_scene does this first
    print(f"[SCRIPTED SCENE] {len(REMAINING)} object(s) on the pretend table")
    return [
        {"id": f"obj_{i}", "waste_class": waste_class,
         "position": {"x": x, "y": y, "z": z},
         "confidence": 1.0, "width_m": 0.05, "height_m": 0.05}
        for i, (waste_class, x, y, z) in enumerate(REMAINING)
    ]


_real_pick = langchain_tools.attempt_pick
_real_place = langchain_tools.attempt_place


def pick_and_remember(*args, **kwargs):
    """Runs the real pick, and remembers what was picked up."""
    target = args[0] if args else kwargs.get("target")
    result = _real_pick(*args, **kwargs)
    if result.get("status") == "success" and target is not None:
        p = target["position"]
        held["item"] = (target["waste_class"], p["x"], p["y"], p["z"])
    return result


def place_and_remove(*args, **kwargs):
    """Runs the real place. After a successful place, the object leaves the table."""
    result = _real_place(*args, **kwargs)
    if result.get("status") == "success":
        if held["item"] in REMAINING:
            REMAINING.remove(held["item"])
            print(f"[SCRIPTED SCENE] {held['item'][0]} removed from the table")
        held["item"] = None
    return result


# Swap in the fakes. Everything else in the agent stays real.
langchain_tools.observe_scene = fake_observe_scene
langchain_tools.attempt_pick = pick_and_remember
langchain_tools.attempt_place = place_and_remove

print("=== Real agent | MATLAB arm | scripted scene ===")
print("Make sure matlab_server is running. Press Ctrl+C here to abort.\n")

# Starts run_agent exactly as 'python -m src.interface.run_agent ...' would,
# using the same command-line arguments you typed after this script's name.
runpy.run_module("src.interface.run_agent", run_name="__main__", alter_sys=True)
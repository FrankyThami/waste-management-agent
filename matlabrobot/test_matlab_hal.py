"""
test_matlab_hal.py - HAL layer (updated for the downward gripper, G4b)
Tests hal.py in MATLAB mode WITHOUT the agent, camera or Gemini.
MATLAB must already be running matlab_server.
Run from the project root:   python matlabrobot\\test_matlab_hal.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(1, HERE)

os.environ["HAL_MODE"] = "matlab"           # must be set BEFORE importing hal
os.environ.setdefault("HAL_ANIMATE", "1")   # "1" = show the arm moving

from src.actions import hal  # noqa: E402
import sim_layout  # noqa: E402

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def as_dict(xyz):
    return {"x": xyz[0], "y": xyz[1], "z": xyz[2]}


_, px, py, pz = next(obj for obj in sim_layout.OBJECTS if obj[0] == "paper")
PICK = {"x": px, "y": py, "z": pz}
BELOW = as_dict(sim_layout.BELOW_FLOOR)
FAR = as_dict(sim_layout.OUT_OF_REACH)

print(f"--- HAL test (mode = {hal.HAL_MODE}) ---")
check("HAL is in MATLAB mode", hal.HAL_MODE == "matlab")
check("reset to the home pose", hal.reset_simulation())

# 1. Can the arm reach every bin?
for waste_class in ["paper", "plastic", "metal", "glass", "organic", "unsorted"]:
    bin_pos = hal.get_bin_position(waste_class)
    check(f"bin '{waste_class}' {bin_pos} is reachable", hal.check_reachable(bin_pos))

# 2. Pick and place (the same steps as tools.attempt_pick and attempt_place)
check("pick spot is reachable", hal.check_reachable(PICK))
check("move to pick spot", hal.simulate_move_to(PICK))
check("close gripper (grasp)", hal.simulate_grasp())
paper_bin = hal.get_bin_position("paper")
check("move to the paper bin", hal.simulate_move_to(paper_bin))
check("open gripper (release)", hal.simulate_release())

# 3. Safety: the floor and the reach limit
check("below-the-floor target is NOT reachable", not hal.check_reachable(BELOW))
check("below-the-floor move is refused", not hal.simulate_move_to(BELOW))
check("out-of-reach target is NOT reachable", not hal.check_reachable(FAR))
check("out-of-reach move is refused", not hal.simulate_move_to(FAR))

# 4. Your simulated grasp failures still work in MATLAB mode
hal.inject_failures(grasp=1)
check("injected grasp failure is reported", not hal.simulate_grasp())

# 5. Stop and reset
check("stop is accepted", hal.simulate_stop())
check("move after stop is refused", not hal.simulate_move_to(PICK))
check("reset clears the stop", hal.reset_simulation())
check("move works again after reset", hal.simulate_move_to(PICK))

hal.reset_simulation()
print(f"\n{passed} passed, {failed} failed")
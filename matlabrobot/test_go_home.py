"""
test_go_home.py - Stage G5b
Checks: go_home, the 'home before observing' hook and its on/off switch, and that
safe_stop (simulate_stop) is a pure halt that never moves the arm.
Also checks that the hover point over every bin and object is reachable.
MATLAB must be running the UPDATED matlab_server (restarted after the edits).
Run from the project root:   python matlabrobot\\test_go_home.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(1, HERE)

os.environ["HAL_MODE"] = "matlab"       # must be set BEFORE importing hal
os.environ["HAL_ANIMATE"] = "0"         # this test checks logic, so no drawing (faster)
os.environ["HAL_HOME_ON_OBSERVE"] = "1"

from src.actions import hal, matlab_hal  # noqa: E402
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


def near(tip, target, tolerance=0.003):
    return all(abs(a - b) < tolerance for a, b in zip(tip, target))


def spot(waste_class):
    return next({"x": x, "y": y, "z": z} for c, x, y, z in sim_layout.OBJECTS if c == waste_class)


def xyz(d):
    return [d["x"], d["y"], d["z"]]


HOME = list(sim_layout.HOME)
TRAVEL = matlab_hal.SAFE_TRAVEL_Z

print(f"--- Go-home test (travel height = {TRAVEL * 1000:.0f} mm) ---")
check("HAL is in MATLAB mode", hal.HAL_MODE == "matlab")
check("reset to the home pose", hal.reset_simulation())

client = matlab_hal._get_client()
state = client.get_state()
check("MATLAB reports the home position", "home" in state and near(state["home"], HOME))

# 1. Is the hover point (travel height) over every bin and object reachable?
for waste_class in ["paper", "plastic", "metal", "glass", "organic", "unsorted"]:
    b = hal.get_bin_position(waste_class)
    check(f"hover point over the {waste_class} bin is reachable",
          hal.check_reachable({"x": b["x"], "y": b["y"], "z": TRAVEL}))
for waste_class in ["paper", "plastic", "organic", "metal"]:
    s = spot(waste_class)
    check(f"hover point over the {waste_class} object is reachable",
          hal.check_reachable({"x": s["x"], "y": s["y"], "z": TRAVEL}))

# Spy on every move command that goes to MATLAB.
calls = []
_real_move = client.move_to


def spy(x, y, z, animate=True):
    calls.append((round(x, 4), round(y, 4), round(z, 4)))
    return _real_move(x, y, z, animate)


client.move_to = spy

# 2. go_home when already at home: succeeds without any motion
check("go_home when already at home succeeds without moving", hal.go_home() and len(calls) == 0)

# 3. A full pick and place, then go home
check("move to the paper object", hal.simulate_move_to(spot("paper")))
check("grasp the paper object", hal.simulate_grasp())
paper_bin = hal.get_bin_position("paper")
check("move to the paper bin", hal.simulate_move_to(paper_bin))
check("release the paper object", hal.simulate_release())
check("go_home from the paper bin succeeds", hal.go_home())
state = client.get_state()
check("fingertips are at the home position", near(state["tip"], HOME))
check("gripper is open at home", state["gripper"] == "open")

# 4. The observation hook: from the plastic object, the arm goes home
check("move to the plastic object", hal.simulate_move_to(spot("plastic")))
check("the observation hook succeeds", hal.go_home_for_observation())
state = client.get_state()
check("fingertips are at home after the hook", near(state["tip"], HOME))

# 5. Switch off: the hook does nothing
hal.HOME_ON_OBSERVE = False
organic = spot("organic")
check("move to the organic object", hal.simulate_move_to(organic))
before = len(calls)
ok = hal.go_home_for_observation()
state = client.get_state()
check("with the switch off, the hook does nothing",
      ok and len(calls) == before and near(state["tip"], xyz(organic)))
hal.HOME_ON_OBSERVE = True

# 6. safe_stop is a PURE HALT
check("safe_stop (simulate_stop) is accepted", hal.simulate_stop())
state = client.get_state()
check("stop is a pure halt: the arm did not move",
      state["stopped"] is True and near(state["tip"], xyz(organic)) and len(calls) == before)
check("go_home is refused while stopped", not hal.go_home())
check("the observation hook reports failure while stopped", not hal.go_home_for_observation())
check("reset clears the stop", hal.reset_simulation())
state = client.get_state()
check("arm is back at the home pose", near(state["tip"], HOME, 0.005))

print(f"\n{passed} passed, {failed} failed")
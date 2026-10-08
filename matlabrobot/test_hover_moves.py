"""
test_hover_moves.py - Stage G5a
Checks that HAL moves are made of lift -> travel -> descend, that sideways travel
only happens at the safe height, and that refused moves leave the arm untouched.
It spies on the commands sent to MATLAB. No agent, no camera, no Gemini.
MATLAB must already be running matlab_server.
Run from the project root:   python matlabrobot\\test_hover_moves.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(1, HERE)

os.environ["HAL_MODE"] = "matlab"       # must be set BEFORE importing hal
os.environ["HAL_ANIMATE"] = "0"         # this test checks logic, so no drawing (faster)

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


def close(a, b, tolerance=0.002):
    return abs(a - b) < tolerance


def spot(waste_class):
    return next({"x": x, "y": y, "z": z} for c, x, y, z in sim_layout.OBJECTS if c == waste_class)


def as_dict(xyz):
    return {"x": xyz[0], "y": xyz[1], "z": xyz[2]}


def near(tip, target, tolerance=0.003):
    return all(close(a, b, tolerance) for a, b in zip(tip, target))


TRAVEL = matlab_hal.SAFE_TRAVEL_Z
HOME = sim_layout.HOME


def travel_is_high(points, start):
    """Every sideways movement must start and end at or above the travel height."""
    previous = start
    for p in points:
        sideways = abs(p[0] - previous[0]) > 0.002 or abs(p[1] - previous[1]) > 0.002
        if sideways and (p[2] < TRAVEL - 0.002 or previous[2] < TRAVEL - 0.002):
            return False
        previous = p
    return True


print(f"--- Hover moves test (travel height = {TRAVEL * 1000:.0f} mm) ---")
check("HAL is in MATLAB mode", hal.HAL_MODE == "matlab")
check("reset to the home pose", hal.reset_simulation())

# Spy on every move command that goes to MATLAB.
client = matlab_hal._get_client()
calls = []
_real_move = client.move_to


def spy(x, y, z, animate=True):
    calls.append((round(x, 4), round(y, 4), round(z, 4)))
    return _real_move(x, y, z, animate)


client.move_to = spy

# 1. Home -> paper object: should be lift, travel, descend
paper = spot("paper")
check("move to the paper object succeeds", hal.simulate_move_to(paper))
three = len(calls) == 3
lift, hover, down = calls[:3] if three else (None, None, None)
check("the move has three segments", three)
check("segment 1: lifts straight up from home to the travel height",
      three and close(lift[0], HOME[0]) and close(lift[1], HOME[1]) and close(lift[2], TRAVEL))
check("segment 2: travels sideways at the travel height",
      three and close(hover[0], paper["x"]) and close(hover[1], paper["y"]) and close(hover[2], TRAVEL))
check("segment 3: descends to the object",
      three and close(down[0], paper["x"]) and close(down[1], paper["y"]) and close(down[2], paper["z"]))

# 2. Grasp, then carry to the paper bin (lifts first, with the object)
check("grasp the paper object", hal.simulate_grasp())
paper_bin = hal.get_bin_position("paper")
check("move to the paper bin succeeds", hal.simulate_move_to(paper_bin))
state = client.get_state()
check("fingertips are at the paper bin",
      near(state["tip"], [paper_bin["x"], paper_bin["y"], paper_bin["z"]]))
check("release the paper object", hal.simulate_release())

# 3. A second object, to make sure the route from a bin to the next object also works
plastic = spot("plastic")
check("move to the plastic object succeeds", hal.simulate_move_to(plastic))
check("grasp the plastic object", hal.simulate_grasp())
plastic_bin = hal.get_bin_position("plastic")
check("move to the plastic bin succeeds", hal.simulate_move_to(plastic_bin))
check("release the plastic object", hal.simulate_release())

check("ALL sideways travel so far happened at the safe height", travel_is_high(calls, HOME))

# 4. Refused moves must leave the arm untouched (no command reaches the arm)
before = len(calls)
ok = hal.simulate_move_to(as_dict(sim_layout.OUT_OF_REACH))
check("out-of-reach move is refused with no motion at all", (not ok) and len(calls) == before)

ok = hal.simulate_move_to(as_dict(sim_layout.BELOW_FLOOR))
check("below-the-floor move is refused with no motion at all", (not ok) and len(calls) == before)

# 5. Stop and reset
check("stop is accepted", hal.simulate_stop())
check("move after stop is refused", not hal.simulate_move_to(paper))
check("reset clears the stop", hal.reset_simulation())
state = client.get_state()
check("arm is back at the home pose", near(state["tip"], list(HOME), 0.005))

print(f"\n{passed} passed, {failed} failed")
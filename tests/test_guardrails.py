"""
agent/tests/test_guardrails.py

Tests the feasibility guardrail with NO camera and NO Gemini calls.

Run from the project root:
    python -m agent.tests.test_guardrails
"""

import sys

from src.actions import hal, langchain_tools as lt, tools
from src.context import state
from src.reliability import guardrails
from src import policy
from src.actions.tools import _estimate_footprint_m

failures = 0


def check(name, condition):
    global failures
    print(("  PASS  " if condition else "  FAIL  ") + name)
    if not condition:
        failures += 1


def obj(idx, waste_class, x, w, h):
    return {"id": f"obj_{idx}", "waste_class": waste_class,
            "position": {"x": x, "y": 0.0, "z": 0.4}, "confidence": 1.0,
            "width_m": w, "height_m": h}


class FakeIntrinsics:
    fx = 1000.0
    fy = 1000.0


LIMIT = policy.MAX_OBJECT_WIDTH_M

print("\n1. The size maths (pixels + distance + focal length -> metres)")
# A box covering the WHOLE 1280x720 image, seen from 0.5 m with a 1000 px focal length:
w, h = _estimate_footprint_m([0, 0, 1000, 1000], 1280, 720, 0.5, FakeIntrinsics())
check("full-width box at 0.5 m is 0.64 m wide", abs(w - 0.64) < 1e-9)
check("full-height box at 0.5 m is 0.36 m tall", abs(h - 0.36) < 1e-9)
w2, _ = _estimate_footprint_m([0, 0, 1000, 500], 1280, 720, 1.0, FakeIntrinsics())
check("half the image width at double the distance is 0.64 m", abs(w2 - 0.64) < 1e-9)

print("\n2. The rule itself")
check("small object is allowed", guardrails.footprint_problem(obj(0, "paper", 0, 0.05, 0.05)) is None)
check("exactly at the limit is allowed", guardrails.footprint_problem(obj(0, "paper", 0, LIMIT, LIMIT)) is None)
check("just over the limit is refused", guardrails.footprint_problem(obj(0, "paper", 0, LIMIT + 0.01, LIMIT + 0.01)) is not None)
check("long thin object is allowed (narrowest side counts)", guardrails.footprint_problem(obj(0, "paper", 0, 0.40, 0.03)) is None)
check("object with no measured size is refused (fails closed)",
      guardrails.footprint_problem({"waste_class": "paper", "position": {"x": 0, "y": 0, "z": 0.4}}) is not None)

print("\n3. Through the tools: an oversized object is refused and skipped for good")
state.new_session()
lt._last_observation.clear()
hal.inject_failures(0, 0)
hal.set_failure_rates(0.0, 0.0)
scene = [obj(0, "paper", 0.10, 0.30, 0.30), obj(1, "plastic", -0.10, 0.05, 0.05)]
lt.observe_scene = lambda: [dict(o) for o in scene]

obs = lt.observe_scene_tool.invoke({})
check("observation shows each object's size", "30 x 30 cm" in obs and "5 x 5 cm" in obs)

arm_moves = []
_real_move = tools.simulate_move_to
tools.simulate_move_to = lambda position: (arm_moves.append(position), _real_move(position))[1]

big = lt.attempt_pick_tool.invoke({"target_id": "obj_0"})
check("oversized object comes back infeasible", big.startswith("infeasible") and "gripper limit" in big)
check("and is skipped immediately (no pointless retry)", "skipped" in big)
check("the arm was never commanded to move for it", arm_moves == [])
again = lt.attempt_pick_tool.invoke({"target_id": "obj_0"})
check("trying it again is refused by the code", again.startswith("rejected"))

small = lt.attempt_pick_tool.invoke({"target_id": "obj_1"})
check("the small object is still picked normally", small.startswith("success"))
check("the arm moved exactly once, for the small object", len(arm_moves) == 1)
tools.simulate_move_to = _real_move

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else str(failures) + ' CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
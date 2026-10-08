"""
agent/tests/test_approval.py

Tests the human-approval gate with NO camera, NO Gemini, and no real
terminal waiting - input() is redirected to canned answers.

Run from the project root:
    python -m agent.tests.test_approval
"""

import builtins
import sys

from src.actions import hal, langchain_tools as lt
from src.context import state
from src.reliability import guardrails
from src import policy

failures = 0


def check(name, condition):
    global failures
    print(("  PASS  " if condition else "  FAIL  ") + name)
    if not condition:
        failures += 1


def obj(idx, waste_class, x, size=0.05):
    return {"id": f"obj_{idx}", "waste_class": waste_class,
            "position": {"x": x, "y": 0.0, "z": 0.4}, "confidence": 1.0,
            "width_m": size, "height_m": size}


def answer(*responses):
    """Feeds canned answers to input(), one per call, in order."""
    it = iter(responses)
    builtins.input = lambda prompt="": next(it)


def new_run(objects):
    state.new_session()
    lt._last_observation.clear()
    hal.inject_failures(0, 0)
    hal.set_failure_rates(0.0, 0.0)
    lt.observe_scene = lambda: [dict(o) for o in objects]
    lt.observe_scene_tool.invoke({})


LIMIT = policy.MAX_OBJECT_WIDTH_M

print("\n1. Which objects trigger the gate")
check("a flagged class always needs approval",
      guardrails.approval_reason(obj(0, "unsorted", 0)) is not None)
check("a small, normal-class object does not",
      guardrails.approval_reason(obj(0, "paper", 0, size=0.03)) is None)
check("an object right at the limit needs approval (borderline)",
      guardrails.approval_reason(obj(0, "paper", 0, size=LIMIT)) is not None)
check("an object comfortably under the borderline band does not",
      guardrails.approval_reason(obj(0, "paper", 0, size=LIMIT * 0.5)) is None)

print("\n2. A denied pick is skipped for good, with no retries wasted")
new_run([obj(0, "unsorted", 0.10)])
answer("n")
r1 = lt.attempt_pick_tool.invoke({"target_id": "obj_0"})
check("denial reports 'declined'", r1.startswith("declined"))
r2 = lt.attempt_pick_tool.invoke({"target_id": "obj_0"})
check("trying again is refused by the code, not re-asked", r2.startswith("rejected") and "skipped" in r2)

print("\n3. An approved pick proceeds normally")
new_run([obj(0, "unsorted", 0.10)])
answer("y")
r3 = lt.attempt_pick_tool.invoke({"target_id": "obj_0"})
check("approval leads to a real pick attempt", r3.startswith("success"))

print("\n4. Approval is remembered - a retry after a failure does NOT ask again")
new_run([obj(0, "unsorted", 0.10)])
hal.inject_failures(grasp=1)
answer("y")  # only ONE answer queued - a second input() call would crash with StopIteration
a = lt.attempt_pick_tool.invoke({"target_id": "obj_0"})
b = lt.attempt_pick_tool.invoke({"target_id": "obj_0"})  # retry after the forced failure
check("first attempt fails (as forced) but was still approved", a.startswith("failed"))
check("retry succeeds without a second approval prompt", b.startswith("success"))

print("\n5. An unanswered prompt (Ctrl+C / closed input) denies by default")
new_run([obj(0, "unsorted", 0.10)])
def boom(prompt=""): raise KeyboardInterrupt
builtins.input = boom
r5 = lt.attempt_pick_tool.invoke({"target_id": "obj_0"})
check("no answer given is treated as a denial", r5.startswith("declined"))

print("\n6. A normal object never triggers the gate at all")
new_run([obj(0, "plastic", 0.10, size=0.03)])
def crash(prompt=""): raise AssertionError("input() should never be called for an ordinary object")
builtins.input = crash
r6 = lt.attempt_pick_tool.invoke({"target_id": "obj_0"})
check("an ordinary small object is picked with no prompt at all", r6.startswith("success"))

print("\n7. An object already OVER the hard limit is never asked about - it's simply infeasible")
def crash2(prompt=""): raise AssertionError("input() should never be called for a way-oversized object")
builtins.input = crash2
new_run([obj(0, "paper", 0.10, size=0.30)])  # 3x the limit
r7 = lt.attempt_pick_tool.invoke({"target_id": "obj_0"})
check("oversized object is rejected as infeasible with no prompt", r7.startswith("infeasible"))

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else str(failures) + ' CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
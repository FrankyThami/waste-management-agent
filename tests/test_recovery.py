"""
agent/tests/test_recovery.py

Tests the recovery rules with NO camera and NO Gemini calls: the scene is
faked and the simulated arm is told exactly when to fail.

Run from the project root:
    python -m agent.tests.test_recovery
"""

import sys

from src.actions import hal, langchain_tools as lt
from src.context import state

failures = 0


def check(name, condition):
    global failures
    print(("  PASS  " if condition else "  FAIL  ") + name)
    if not condition:
        failures += 1


def obj(idx, waste_class, x, y, size=0.05):
    return {"id": f"obj_{idx}", "waste_class": waste_class,
            "position": {"x": x, "y": y, "z": 0.4}, "confidence": 1.0,
            "width_m": size, "height_m": size}


def new_run(objects):
    """Fresh session, fresh fake scene, no failures switched on."""
    state.new_session()
    lt._last_observation.clear()
    hal.inject_failures(0, 0)
    hal.set_failure_rates(0.0, 0.0)
    lt.observe_scene = lambda: [dict(o) for o in objects]


def call(tool, **args):
    return tool.invoke(args)


print("\n1. A pick that fails twice: retry once, then skip")
new_run([obj(0, "plastic", 0.10, 0.0)])
call(lt.observe_scene_tool)
hal.inject_failures(grasp=2)
r1 = call(lt.attempt_pick_tool, target_id="obj_0")
r2 = call(lt.attempt_pick_tool, target_id="obj_0")
r3 = call(lt.attempt_pick_tool, target_id="obj_0")
check("first failure allows one retry", r1.startswith("failed") and "retry" in r1)
check("second failure skips the object", r2.startswith("failed") and "skipped" in r2)
check("a third attempt is refused by the code", r3.startswith("rejected") and "skipped" in r3)

print("\n2. Re-observing: the skipped object is recognised even with measurement jitter")
lt.observe_scene = lambda: [obj(0, "plastic", 0.12, 0.01)]  # ~2 cm away from before
obs = call(lt.observe_scene_tool)
check("skipped object is filtered out", "No objects detected" in obs)
check("the agent is told it was ignored", "previously skipped" in obs)

print("\n3. A pick that fails once, then succeeds (a retry that works)")
new_run([obj(0, "plastic", 0.10, 0.0), obj(1, "paper", -0.10, 0.0)])
call(lt.observe_scene_tool)
hal.inject_failures(grasp=1)
a = call(lt.attempt_pick_tool, target_id="obj_0")
b = call(lt.attempt_pick_tool, target_id="obj_0")
check("first attempt fails", a.startswith("failed"))
check("retry succeeds", b.startswith("success"))

print("\n4. Only one object can be held at a time")
c = call(lt.attempt_pick_tool, target_id="obj_1")
check("picking while holding is refused", c.startswith("rejected") and "holding" in c)
d = call(lt.attempt_place_tool, waste_class="plastic")
check("placing the held object succeeds", d.startswith("success"))
e = call(lt.attempt_pick_tool, target_id="obj_1")
check("the gripper is free again afterwards", e.startswith("success"))

print("\n5. Placing with an empty gripper")
new_run([obj(0, "plastic", 0.10, 0.0)])
call(lt.observe_scene_tool)
f = call(lt.attempt_place_tool, waste_class="plastic")
check("place without holding is refused", f.startswith("rejected") and "not holding" in f)

print("\n6. A place that fails twice aborts the whole task")
new_run([obj(0, "plastic", 0.10, 0.0), obj(1, "paper", -0.10, 0.0)])
call(lt.observe_scene_tool)
call(lt.attempt_pick_tool, target_id="obj_0")
hal.inject_failures(release=2)
g1 = call(lt.attempt_place_tool, waste_class="plastic")
g2 = call(lt.attempt_place_tool, waste_class="plastic")
check("first failed place allows one retry", g1.startswith("failed") and "again" in g1)
check("second failed place aborts the task", g2.startswith("failed") and "ABORTED" in g2)
check("observe is refused after an abort", call(lt.observe_scene_tool).startswith("rejected"))
check("pick is refused after an abort", call(lt.attempt_pick_tool, target_id="obj_1").startswith("rejected"))
check("safe_stop still works after an abort", call(lt.safe_stop_tool, reason="done").startswith("stopped"))

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else str(failures) + ' CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
"""
test_tools_matlab.py - tools layer (updated for the downward gripper, G4b)
Runs the REAL tools (observe_scene_tool, attempt_pick_tool, attempt_place_tool,
safe_stop_tool) in MATLAB mode. Only the camera function is swapped for a fake
that returns the objects from sim_layout.py. No camera, no Gemini, no agent loop.
MATLAB must already be running matlab_server.
Run from the project root:   python matlabrobot\\test_tools_matlab.py
To watch the arm move:   $env:HAL_ANIMATE = "1"   (slower)
"""
import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(1, HERE)

os.environ["HAL_MODE"] = "matlab"            # must be set BEFORE importing the tools
os.environ.setdefault("HAL_ANIMATE", "0")    # "0" = faster; set to "1" to watch

from src.actions import hal, matlab_hal, langchain_tools  # noqa: E402
from src.actions.langchain_tools import (  # noqa: E402
    observe_scene_tool, attempt_pick_tool, attempt_place_tool, safe_stop_tool,
)
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


def says(result, word):
    """True if the tool's result mentions this word (works for dict or text results)."""
    return word in str(result).lower()


def near(tip, target, tolerance=0.003):
    return all(abs(a - b) < tolerance for a, b in zip(tip, target))


def arm_state():
    return matlab_hal._get_client().get_state()


def fake_observe_scene():
    """Stands in for the camera + Gemini. Same return type as the real observe_scene()."""
    return [
        {"id": f"obj_{i}", "waste_class": waste_class,
         "position": {"x": x, "y": y, "z": z},
         "confidence": 1.0, "width_m": 0.05, "height_m": 0.05}
        for i, (waste_class, x, y, z) in enumerate(sim_layout.OBJECTS)
    ]


def pick_spot(waste_class):
    return next([x, y, z] for c, x, y, z in sim_layout.OBJECTS if c == waste_class)


def bin_spot(waste_class):
    b = hal.get_bin_position(waste_class)
    return [b["x"], b["y"], b["z"]]


print(f"--- Tools test in MATLAB mode (HAL_MODE = {hal.HAL_MODE}) ---")
check("HAL is in MATLAB mode", hal.HAL_MODE == "matlab")
check("reset to the home pose", hal.reset_simulation())

# Run the REAL observe_scene_tool, but with the camera function swapped for the fake.
with mock.patch.object(langchain_tools, "observe_scene", fake_observe_scene):
    scene_text = observe_scene_tool.invoke({})
print("   observe ->", scene_text)

scene = langchain_tools._last_observation
check("scripted scene registered (all objects)", len(scene) == len(sim_layout.OBJECTS))
if len(scene) != len(sim_layout.OBJECTS):
    print("\nThe scripted scene was not registered, so the rest cannot run.")
    print("Send me this output.")
    raise SystemExit(1)

id_of = {obj["waste_class"]: obj_id for obj_id, obj in scene.items()}

# 1. Paper: normal pick and place
result = attempt_pick_tool.invoke({"target_id": id_of["paper"]})
print("   pick paper ->", result)
check("pick paper succeeds", says(result, "success"))
state = arm_state()
check("after pick: gripper closed, fingertips at the pick spot",
      state["gripper"] == "closed" and near(state["tip"], pick_spot("paper")))

result = attempt_place_tool.invoke({"waste_class": "paper"})
print("   place paper ->", result)
check("place paper succeeds", says(result, "success"))
state = arm_state()
check("after place: gripper open, fingertips at the paper bin",
      state["gripper"] == "open" and near(state["tip"], bin_spot("paper")))

# 2. Plastic: normal pick and place
result = attempt_pick_tool.invoke({"target_id": id_of["plastic"]})
print("   pick plastic ->", result)
check("pick plastic succeeds", says(result, "success"))
result = attempt_place_tool.invoke({"waste_class": "plastic"})
print("   place plastic ->", result)
check("place plastic succeeds", says(result, "success"))

# 3. Glass: out of reach, must be refused
result = attempt_pick_tool.invoke({"target_id": id_of["glass"]})
print("   pick glass ->", result)
check("unreachable object is refused", says(result, "unreachable") and not says(result, "success"))

# 4. Organic: simulated grasp failure, then a retry
hal.inject_failures(grasp=1)
result = attempt_pick_tool.invoke({"target_id": id_of["organic"]})
print("   pick organic (injected failure) ->", result)
check("injected grasp failure is reported", says(result, "failed") and not says(result, "success"))
result = attempt_pick_tool.invoke({"target_id": id_of["organic"]})
print("   retry organic ->", result)
check("retry after the failure succeeds", says(result, "success"))
result = attempt_place_tool.invoke({"waste_class": "organic"})
print("   place organic ->", result)
check("place organic succeeds", says(result, "success"))

# 5. Metal: emergency stop
result = safe_stop_tool.invoke({"reason": "tools test stop"})
print("   safe_stop ->", result)
check("safe_stop reports stopped", says(result, "stopped"))
result = attempt_pick_tool.invoke({"target_id": id_of["metal"]})
print("   pick metal after stop ->", result)
check("pick is refused after stop", not says(result, "success"))

check("reset clears the stop", hal.reset_simulation())
state = arm_state()
check("arm is back at the home pose, gripper open",
      state["gripper"] == "open" and near(state["tip"], list(sim_layout.HOME), 0.005))

print(f"\n{passed} passed, {failed} failed")
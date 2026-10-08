"""
test_arm_link.py - bridge layer (updated for the downward gripper, G4b)
Tests the MATLAB commands directly. MATLAB must be running matlab_server.
Positions are FINGERTIP positions (see sim_layout.py).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from matlab_bridge import MatlabRobotClient, MatlabBridgeError  # noqa: E402
import sim_layout  # noqa: E402

passed = 0
failed = 0

HOME = list(sim_layout.HOME)
PICK = [0.45, 0.00, sim_layout.PICK_Z]       # where the paper object sits
LIFT = [0.45, 0.00, 0.20]                    # straight above it
BELOW = list(sim_layout.BELOW_FLOOR)
FAR = list(sim_layout.OUT_OF_REACH)


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {name} {detail}")
    else:
        failed += 1
        print(f"FAIL  {name} {detail}")


def near(tip, target, tolerance=0.003):
    return all(abs(a - b) < tolerance for a, b in zip(tip, target))


print("--- Arm link test (gripper down) ---")
try:
    with MatlabRobotClient() as robot:
        robot.reset()    # always start from the home pose

        reply = robot.ping()
        check("ping", reply.get("status") == "ok", f"-> {reply.get('reason')}")

        state = robot.get_state()
        check("start state: home pose, gripper open",
              near(state["tip"], HOME) and state["gripper"] == "open" and state["stopped"] is False,
              f"-> {state['tip']}, {state['gripper']}")

        reply = robot.check_position(*PICK)
        check("check_position (pick spot) says ok", reply.get("status") == "ok",
              f"-> {reply.get('status')}")

        reply = robot.check_position(*BELOW)
        check("check_position (below the floor) is blocked",
              reply.get("status") == "blocked" and "floor" in reply.get("reason", ""),
              f"-> {reply.get('reason')}")

        reply = robot.check_position(*FAR)
        check("check_position (out of reach) is blocked", reply.get("status") == "blocked",
              f"-> {reply.get('reason')}")

        reply = robot.move_to(*PICK, animate=False)
        check("move to the pick spot", reply.get("status") == "done", f"-> {reply.get('status')}")

        reply = robot.move_to(*LIFT, animate=True)
        check("lift straight up (animated)", reply.get("status") == "done",
              f"-> {reply.get('status')}")

        state = robot.get_state()
        check("arm REMEMBERS its position", near(state["tip"], LIFT), f"-> {state['tip']}")

        reply = robot.set_gripper("closed")
        check("close gripper", reply.get("status") == "done", f"-> {reply.get('reason')}")

        reply = robot.move_to(*BELOW, animate=False)
        state = robot.get_state()
        check("below-the-floor move blocked and arm did not move",
              reply.get("status") == "blocked" and near(state["tip"], LIFT),
              f"-> {reply.get('reason')}")

        robot.stop()
        reply = robot.move_to(*PICK, animate=False)
        check("move after stop is refused", reply.get("status") == "blocked",
              f"-> {reply.get('reason')}")

        robot.reset()
        state = robot.get_state()
        check("reset: home pose, gripper open, not stopped",
              near(state["tip"], HOME) and state["gripper"] == "open" and state["stopped"] is False,
              f"-> {state['tip']}")

except MatlabBridgeError as error:
    print(f"Could not talk to MATLAB: {error}")
    raise SystemExit(1)

print(f"\n{passed} passed, {failed} failed")
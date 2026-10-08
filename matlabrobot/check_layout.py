"""
check_layout.py - Stage G4a
Checks the simulated table layout (sim_layout.py + config/matlab_bin_positions.json)
against the MATLAB arm: is every object and bin reachable, does the floor rule work,
and can the arm make the object -> bin routes? No agent, no HAL, no animation (fast).
MATLAB must be running the NEW matlab_server (started after 'clear classes').
Run from the project root:   python matlabrobot\\check_layout.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from matlab_bridge import MatlabRobotClient, MatlabBridgeError  # noqa: E402
import sim_layout  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(HERE), "config", "matlab_bin_positions.json")
with open(CONFIG_PATH, "r") as f:
    BINS = json.load(f)["bins"]

problems = 0


def verdict(ok, text):
    global problems
    if not ok:
        problems += 1
    print(f"{'OK  ' if ok else 'FAIL'} {text}")
    return ok


def short(reply):
    return f"{reply.get('status')} | {reply.get('reason')}"


print("--- Layout check ---")
try:
    with MatlabRobotClient() as robot:
        robot.reset()

        print("\n--- Bins (release positions) ---")
        for name, (x, y, z) in BINS.items():
            reply = robot.check_position(x, y, z)
            verdict(reply.get("status") == "ok",
                    f"bin {name:8s} [{x:.2f} {y:+.2f} {z:.3f}] -> {short(reply)}")

        print("\n--- Objects (pick positions) ---")
        for waste_class, x, y, z in sim_layout.OBJECTS:
            reply = robot.check_position(x, y, z)
            reachable = reply.get("status") == "ok"
            on_purpose = waste_class in sim_layout.UNREACHABLE_ON_PURPOSE
            note = "  (unreachable on purpose)" if on_purpose else ""
            verdict(reachable != on_purpose,
                    f"object {waste_class:8s} [{x:.2f} {y:+.2f} {z:.3f}] -> {short(reply)}{note}")

        print("\n--- Floor rule ---")
        x, y, z = sim_layout.BELOW_FLOOR
        reply = robot.check_position(x, y, z)
        verdict(reply.get("status") == "blocked",
                f"below the floor [{x:.2f} {y:+.2f} {z:.3f}] -> {short(reply)}")

        print("\n--- Routes: object -> bin (no animation, takes a little while) ---")
        for waste_class, x, y, z in sim_layout.OBJECTS:
            if waste_class in sim_layout.UNREACHABLE_ON_PURPOSE:
                continue
            if waste_class not in BINS:
                verdict(False, f"{waste_class}: no bin position configured")
                continue

            reply = robot.move_to(x, y, z, animate=False)
            if not verdict(reply.get("status") == "done",
                           f"{waste_class:8s} go to the object  -> {short(reply)}"):
                robot.reset()          # start the next route clean
                continue
            robot.set_gripper("closed")

            bx, by, bz = BINS[waste_class]
            reply = robot.move_to(bx, by, bz, animate=False)
            if not verdict(reply.get("status") == "done",
                           f"{waste_class:8s} carry to its bin  -> {short(reply)}"):
                robot.reset()
                continue
            robot.set_gripper("open")

        robot.reset()

except MatlabBridgeError as error:
    print(f"Could not talk to MATLAB: {error}")
    raise SystemExit(1)

print(f"\nLayout check finished: {problems} problem(s)")
raise SystemExit(1 if problems else 0)
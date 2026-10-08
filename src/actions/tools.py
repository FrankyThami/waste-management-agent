"""
agent/tools.py

The 4 tools the agent can call: observe_scene(), attempt_pick(),
attempt_place() and safe_stop(). Plain Python - no LangChain in here.
"""

from src import policy
from src.actions.observation import observe
from src.reliability import guardrails
from src.schemas import DetectedObject, ToolResult
from src.actions.hal import check_reachable, simulate_move_to, simulate_grasp, simulate_release, simulate_stop, go_home_for_observation


# ---------------------------------------------------------------------
# observe_scene()
# ---------------------------------------------------------------------

def observe_scene(remember: bool = True):
    """
    Parks the arm at the home pose (out of the camera's view), then looks at
    the table once. Returns what is there - waste objects and bins, with
    positions and sizes - exactly as described in observation.observe().
    remember=False is used for re-check looks: they do not replace the picture
    that "what changed" is measured against.
    """
    go_home_for_observation()
    return observe(remember=remember)


# ---------------------------------------------------------------------
# attempt_pick()
# ---------------------------------------------------------------------

def _localize(target):
    """Step 1: read the object's position (already computed by observe_scene)."""
    return target["position"]


def _feasibility_problem(target):
    """Step 2: why this object can't be gripped, or None if it can. Rules live in guardrails.py."""
    return guardrails.footprint_problem(target)


def _check_reachability(position):
    """Step 3: can the arm's workspace actually reach this point?"""
    return check_reachable(position)


def _execute_pick(position):
    """Step 4: move to the object and close the gripper. False if either step fails."""
    return simulate_move_to(position) and simulate_grasp()


def _verify_pick():
    """
    Step 5: confirm the pick actually worked.
    Placeholder for now - always True in simulation. Real verification
    (camera recheck or gripper feedback) plugs in here once hardware exists.
    """
    return True


def attempt_pick(target: DetectedObject) -> ToolResult:
    """
    Attempts to pick up one detected object. Chains localize -> feasibility
    -> reachability -> pick -> verify internally, so the agent makes ONE
    decision (which object) instead of five.
    """
    position = _localize(target)

    problem = _feasibility_problem(target)
    if problem:
        return {"status": "infeasible", "reason": f"{target['waste_class']} object refused: {problem}"}

    if not _check_reachability(position):
        return {"status": "unreachable", "reason": f"Position {position} is outside the arm's workspace"}

    if not _execute_pick(position):
        return {"status": "failed", "reason": "Pick execution failed"}

    if not _verify_pick():
        return {"status": "failed", "reason": "Pick could not be verified"}

    return {"status": "success", "reason": f"Picked {target['waste_class']} object at {position}"}


# ---------------------------------------------------------------------
# attempt_place()
# ---------------------------------------------------------------------

def _release_position(bin_info):
    """Step 1: where the fingertips let go - above the bin's measured rim, by a safety clearance."""
    rim = bin_info["position"]
    return {"x": rim["x"], "y": rim["y"], "z": round(rim["z"] + policy.RELEASE_CLEARANCE_M, 4)}


def _execute_place(position):
    """Step 3: move to the release point and open the gripper. False if either step fails."""
    return simulate_move_to(position) and simulate_release()


def _verify_placement():
    """
    Step 4: confirm the object actually landed in the bin.
    Placeholder for now - always True in simulation. Real verification
    (a quick re-observe of the bin) plugs in here once hardware exists.
    """
    return True


def attempt_place(bin_info: dict, waste_class: str) -> ToolResult:
    """
    Attempts to release the held object over a bin that was seen in the scene.
    bin_info is one bin from observe_scene (its colour and measured rim position).
    Chains release point -> reachability -> move+release -> verify internally.
    """
    colour = bin_info["colour"]
    position = _release_position(bin_info)

    if not check_reachable(position):
        return {"status": "unreachable",
                "reason": f"The release point above the {colour} bin {position} is outside the arm's workspace"}

    if not _execute_place(position):
        return {"status": "failed", "reason": "Place execution failed"}

    if not _verify_placement():
        return {"status": "failed", "reason": "Placement could not be verified"}

    return {"status": "success",
            "reason": f"Released the {waste_class} object over the {colour} bin at {position}"}


# ---------------------------------------------------------------------
# safe_stop()
# ---------------------------------------------------------------------

def safe_stop(reason: str) -> ToolResult:
    """
    Immediately halts the arm and ends the task. Callable from anywhere,
    any time - not just at a normal finish, but mid-pick or mid-place too
    if something looks wrong.
    """
    simulate_stop()
    print(f"[safe_stop] Task halted: {reason}")
    return {"status": "stopped", "reason": reason}
"""
agent/guardrails.py

Hard safety rules, enforced in code. The model cannot skip or override
these - it only ever sees their results.
"""

from src import policy


def _narrowest_m(target):
    """The smaller of the object's two visible dimensions, or None if never measured."""
    width, height = target.get("width_m"), target.get("height_m")
    if width is None or height is None:
        return None
    return min(width, height)


def footprint_problem(target):
    """
    Why this object can't be gripped, or None if it can.

    Assumption: the gripper can approach from any direction, so what matters
    is the object's NARROWEST visible side (a long thin item can be pinched
    across its width; a big flat box cannot). Thickness and weight cannot be
    seen from above, so they are NOT checked here.

    Fails closed: an object whose size was never measured is refused.
    """
    narrowest = _narrowest_m(target)
    if narrowest is None:
        return "the object's size was not measured, so it can't be checked against the gripper limit"
    if narrowest > policy.MAX_OBJECT_WIDTH_M:
        return (f"the object is about {narrowest * 100:.0f} cm across at its narrowest, "
                f"wider than the {policy.MAX_OBJECT_WIDTH_M * 100:.0f} cm gripper limit")
    return None


def approval_reason(target):
    """
    Why this pick needs a human's OK before proceeding, or None if it
    doesn't. Two triggers: a flagged waste class (always), or a size
    close enough to the gripper limit to warrant a second opinion even
    though it technically passed footprint_problem().
    """
    if target["waste_class"] in policy.APPROVAL_REQUIRED_CLASSES:
        return f"objects classified as '{target['waste_class']}' always require approval"

    narrowest = _narrowest_m(target)
    if narrowest is None:
        return None  # no size measured - footprint_problem() will reject it outright, nothing to approve
    threshold = policy.MAX_OBJECT_WIDTH_M * policy.BORDERLINE_SIZE_FRACTION
    if threshold <= narrowest <= policy.MAX_OBJECT_WIDTH_M:
        return (f"the object is close to the gripper limit "
                f"({narrowest * 100:.0f} cm, limit is {policy.MAX_OBJECT_WIDTH_M * 100:.0f} cm)")
    return None  # either comfortably small, or already over the limit (infeasible - not a case to approve)


def ask_human_approval(target, reason):
    """
    Pauses and asks a human whether to proceed. Blocks until answered -
    the agent cannot get around this by choosing a different tool, because
    the check runs before the pick itself, inside attempt_pick_tool.
    Fails closed: no answer (Ctrl+C, closed input) counts as a denial.
    """
    print("\n=== Approval needed ===")
    print(f"Reason: {reason}")
    print(f"Object: {target['waste_class']} at {target['position']}, "
          f"about {target['width_m'] * 100:.0f} x {target['height_m'] * 100:.0f} cm")
    try:
        while True:
            raw = input("Approve this pick? (y/n): ").strip().lower()
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            print("  Please answer y or n.")
    except (KeyboardInterrupt, EOFError):
        print("\nNo answer given - denying by default (fail closed).")
        return False

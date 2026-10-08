"""
stale_guard.py - is the thing the arm is about to touch still where the agent believes? (Stage 3.2)

The agent chooses an object or a bin using positions from its LAST observation. Before the arm
moves, a fresh look is taken and THAT ONE target is compared with what the agent believes.
Only a yes/no and plain facts come out of here; what to do next is still Gemini's decision.
Safety rule: if the target cannot be verified, the arm does not move.
"""

from src.context.scene_diff import compare_observations

# kind -> (group name inside an observation, the label that identifies the target)
_KINDS = {
    "object": ("objects", "waste_class"),
    "bin": ("bins", "colour"),
}


def _xy(position):
    return f"x={position['x']:.3f}, y={position['y']:.3f}"


def _name(kind, believed):
    if kind == "object":
        return f"the {believed['waste_class']} object {believed['id']}"
    return f"the {believed['colour']} bin"


def _refuse(fact, outdated=True):
    """Builds the refusal. 'outdated' adds that the agent's earlier picture is no longer right."""
    print(f"[guard] REFUSED: {fact}")
    tail = " The arm did not move."
    if outdated:
        tail += " What you saw earlier is out of date."
    return {"ok": False, "message": f"refused: {fact}{tail}"}


def _verify(kind, believed, fresh):
    """
    believed: the object/bin dictionary the agent holds from its last observation.
    fresh:    a new result from observe().
    Returns {"ok": True,  "target": the fresh entry (keeping the agent's own id)}
         or {"ok": False, "message": plain-English facts for the agent}.
    """
    group, _label_key = _KINDS[kind]
    name = _name(kind, believed)

    if fresh.get("error"):
        return _refuse(f"{name} could not be re-checked because the vision step failed ({fresh['error']}).",
                       outdated=False)

    # Reuse the Stage 3.1 comparison, with the believed target as the whole "previous look".
    previous = {"objects": [], "bins": []}
    previous[group] = [believed]
    report = compare_observations(previous, fresh, 0.0)[group]

    if report["unchanged"]:
        entry = report["unchanged"][0]
        fresh_entry = next(item for item in fresh[group] if item["id"] == entry["id"])
        print(f"[guard] OK: {name} is where it was seen (shifted {entry['shift_m'] * 1000:.0f} mm).")
        return {"ok": True, "target": {**fresh_entry, "id": believed["id"]}}

    if report["moved"]:
        entry = report["moved"][0]
        return _refuse(f"{name} is no longer where it was seen: it was at {_xy(entry['from'])} "
                       f"and is now at {_xy(entry['to'])}, {entry['shift_m'] * 100:.0f} cm away.")

    if report["reclassified"]:
        entry = report["reclassified"][0]
        return _refuse(f"{name} is at the same spot but is now labelled '{entry['now']}' "
                       f"instead of '{entry['was']}'.")

    return _refuse(f"{name} can no longer be seen at its last position ({_xy(believed['position'])}); "
                   "it may have been removed or covered.")


def check_object(believed, fresh):
    """Is this waste object still where the agent last saw it?"""
    return _verify("object", believed, fresh)


def check_bin(believed, fresh):
    """Is this bin still where the agent last saw it?"""
    return _verify("bin", believed, fresh)
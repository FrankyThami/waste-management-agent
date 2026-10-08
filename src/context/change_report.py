"""
change_report.py - turns the change report from scene_diff.py into plain sentences for the agent (Stage 3.3).

Facts only: what moved, vanished or appeared since the agent's previous look.
Changes the agent caused itself (objects it picked up or placed) are left out, so the agent is
not "surprised" by its own work - on the real arm a picked object really does vanish from the table.
"""


def _xy(position):
    return f"x={position['x']:.3f}, y={position['y']:.3f}"


def describe_changes(changes, is_own_work):
    """
    changes:     the "changes" field from observe() (None on the very first look).
    is_own_work: function(position) -> True if that position is where the agent itself picked
                 something up from or placed it (comes from state.py).
    Returns a list of text lines ([] on the very first look).
    """
    if changes is None:
        return []

    lines = []
    hidden = 0   # changes explained by the agent's own pick-ups and placements
    objects, bins = changes["objects"], changes["bins"]

    for e in objects["moved"]:
        if is_own_work(e["from"]):
            hidden += 1
            continue
        lines.append(f"- MOVED: the {e['waste_class']} object (was {e['was_id']}, now {e['id']}) was at "
                     f"{_xy(e['from'])} and is now at {_xy(e['to'])}, {e['shift_m'] * 100:.0f} cm away.")
    for e in objects["missing"]:
        if is_own_work(e["last_position"]):
            hidden += 1
            continue
        lines.append(f"- GONE: the {e['waste_class']} object {e['was_id']} (last seen at "
                     f"{_xy(e['last_position'])}) can no longer be seen.")
    for e in objects["new"]:
        if is_own_work(e["position"]):
            hidden += 1
            continue
        lines.append(f"- NEW: a {e['waste_class']} object ({e['id']}) has appeared at {_xy(e['position'])}.")
    for e in objects["reclassified"]:
        lines.append(f"- RELABELLED: {e['was_id']} is now {e['id']}; it was seen as '{e['was']}' "
                     f"and is now seen as '{e['now']}' (same spot).")

    for e in bins["moved"]:
        lines.append(f"- BIN MOVED: the {e['colour']} bin was at {_xy(e['from'])} and is now at "
                     f"{_xy(e['to'])}, {e['shift_m'] * 100:.0f} cm away.")
    for e in bins["missing"]:
        lines.append(f"- BIN GONE: the {e['colour']} bin (last seen at {_xy(e['last_position'])}) "
                     "can no longer be seen.")
    for e in bins["new"]:
        lines.append(f"- NEW BIN: a {e['colour']} bin has appeared at {_xy(e['position'])}.")
    for e in bins["reclassified"]:
        lines.append(f"- BIN RELABELLED: the bin at the same spot was seen as '{e['was']}' "
                     f"and is now seen as '{e['now']}'.")

    seconds = changes["seconds_since_previous_look"]
    if not lines:
        if hidden:
            return [f"Nothing else has changed since your previous look ({seconds:.0f} s earlier), "
                    "apart from objects you handled yourself."]
        return [f"Nothing has changed since your previous look ({seconds:.0f} s earlier)."]
    return [f"Changes since your previous look ({seconds:.0f} s earlier; "
            "ids are renumbered every look, 'was' shows the old id):"] + lines
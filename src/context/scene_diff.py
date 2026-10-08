"""
scene_diff.py - compares two looks at the table and reports what changed (Stage 3.1).

Only facts, no decisions: which objects and bins are unchanged, moved, missing or new.
The agent (Gemini) reads this report and decides what to do about it.
Items without a measured position are not compared (observe() lists them as "unmeasured").
"""

import math

# How far (metres) an item's centre may drift between two looks and still count as "not moved".
# Gemini's boxes jitter a little from call to call, so this must be bigger than that noise.
# These are starting values - we tune them from the live test.
OBJECT_MOVED_THRESHOLD_M = 0.02
BIN_MOVED_THRESHOLD_M = 0.03

# A "missing" item and a "new" item this close together are the SAME item that Gemini
# labelled differently this time (e.g. plastic -> glass), not one that vanished and another that appeared.
RECLASSIFY_DISTANCE_M = 0.03


def _distance_m(a, b):
    """Horizontal distance between two positions (x and y only; z is just depth noise on a flat table)."""
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _round_position(position):
    return {axis: round(value, 3) for axis, value in position.items()}


def _match_nearest(previous_items, current_items, allowed):
    """
    Pairs items from the previous look with items from the current look, closest pairs first.
    allowed(prev, cur) says whether two items may be paired at all.
    Returns (pairs, left_over_previous, left_over_current); pairs = [(prev, cur, distance_m), ...]
    """
    candidates = sorted(
        (_distance_m(p["position"], c["position"]), pi, ci)
        for pi, p in enumerate(previous_items)
        for ci, c in enumerate(current_items)
        if allowed(p, c)
    )
    used_previous, used_current, pairs = set(), set(), []
    for distance, pi, ci in candidates:
        if pi in used_previous or ci in used_current:
            continue
        used_previous.add(pi)
        used_current.add(ci)
        pairs.append((previous_items[pi], current_items[ci], distance))
    left_previous = [p for i, p in enumerate(previous_items) if i not in used_previous]
    left_current = [c for i, c in enumerate(current_items) if i not in used_current]
    return pairs, left_previous, left_current


def _compare_group(previous_items, current_items, label_key, moved_threshold_m):
    """Compares objects (label_key='waste_class') or bins (label_key='colour')."""
    report = {"moved": [], "reclassified": [], "missing": [], "new": [], "unchanged": []}

    # Round 1: same label (same class / same colour), closest first.
    pairs, left_previous, left_current = _match_nearest(
        previous_items, current_items, lambda p, c: p[label_key] == c[label_key]
    )
    for prev, cur, distance in pairs:
        entry = {"id": cur["id"], "was_id": prev["id"], label_key: cur[label_key],
                 "shift_m": round(distance, 3)}
        if distance > moved_threshold_m:
            entry["from"] = _round_position(prev["position"])
            entry["to"] = _round_position(cur["position"])
            report["moved"].append(entry)
        else:
            report["unchanged"].append(entry)

    # Round 2: leftovers that sit on the same spot but got a different label.
    cross_pairs, still_previous, still_current = _match_nearest(
        left_previous, left_current, lambda p, c: True
    )
    for prev, cur, distance in cross_pairs:
        if distance <= RECLASSIFY_DISTANCE_M:
            report["reclassified"].append({
                "id": cur["id"], "was_id": prev["id"],
                "was": prev[label_key], "now": cur[label_key], "shift_m": round(distance, 3),
            })
        else:
            still_previous.append(prev)
            still_current.append(cur)

    # Whatever is still unpaired: gone from the table, or newly on it.
    for prev in still_previous:
        report["missing"].append({"was_id": prev["id"], label_key: prev[label_key],
                                  "last_position": _round_position(prev["position"])})
    for cur in still_current:
        report["new"].append({"id": cur["id"], label_key: cur[label_key],
                              "position": _round_position(cur["position"])})
    return report


def compare_observations(previous: dict, current: dict, seconds_between: float) -> dict:
    """
    previous / current: dictionaries from observe() (they need "objects" and "bins").
    Returns the change report the agent receives.
    """
    objects = _compare_group(previous["objects"], current["objects"],
                             "waste_class", OBJECT_MOVED_THRESHOLD_M)
    bins = _compare_group(previous["bins"], current["bins"],
                          "colour", BIN_MOVED_THRESHOLD_M)
    anything_changed = any(
        group[status]
        for group in (objects, bins)
        for status in ("moved", "reclassified", "missing", "new")
    )
    return {
        "seconds_since_previous_look": seconds_between,
        "anything_changed": anything_changed,
        "objects": objects,
        "bins": bins,
    }
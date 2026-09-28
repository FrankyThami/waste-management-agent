"""
agent/state.py

Task state: tracks the status of each object from the most recent
observe_scene() call, so progress is tracked as real data - not just
something the model has to remember correctly on its own.

Repopulated fresh every time observe_scene_tool runs, since a new
observation reflects the current real scene - once an object is
physically removed, there's nothing left to track from before.
"""
import math
_done_positions = []  # positions of objects already successfully placed

_objects = {}       # id -> {"waste_class": ..., "status": ..., "attempts": int}
_holding_id = None  # id of the object currently held, if any


def reset(detected_objects):
    """Called by observe_scene_tool: replaces tracked objects with a fresh batch, all pending."""
    global _holding_id
    _objects.clear()
    _holding_id = None
    for obj in detected_objects:
        _objects[obj["id"]] = {
            "waste_class": obj["waste_class"],
            "position": obj["position"],
            "status": "pending",
            "attempts": 0,
        }


def record_attempt(obj_id):
    """Called whenever attempt_pick_tool runs on an object - Day 3's retry limit will read this count."""
    if obj_id in _objects:
        _objects[obj_id]["attempts"] += 1


def set_holding(obj_id):
    """Called after a successful pick - remembers which object is now being held."""
    global _holding_id
    _holding_id = obj_id


def mark_holding_done():
    """Called after a successful place - marks whichever object was held as done."""
    global _holding_id
    if _holding_id in _objects:
        _objects[_holding_id]["status"] = "done"
        _done_positions.append(_objects[_holding_id]["position"])
    _holding_id = None

def pending_ids():
    """IDs not yet marked done."""
    return [obj_id for obj_id, info in _objects.items() if info["status"] != "done"]


def summary():
    """A short line for the agent to see what's left, in plain language."""
    pending = pending_ids()
    if not pending:
        return "No pending objects remain in this observation."
    lines = [f"{obj_id} ({_objects[obj_id]['waste_class']})" for obj_id in pending]
    return f"{len(pending)} object(s) still pending: " + ", ".join(lines)
def is_pending(obj_id):
    """False if the id is unknown, or already marked done - used to block re-picking a finished object."""
    return obj_id in _objects and _objects[obj_id]["status"] != "done"

def _distance(p1, p2):
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2 + (p1["z"] - p2["z"]) ** 2)


def filter_out_completed(detected_objects, tolerance=0.05):
    """
    Removes any detected object whose position closely matches one
    already successfully placed (within `tolerance` metres).

    Needed because, without a real robot yet, a "placed" object never
    actually leaves the camera's view - the next observation would
    otherwise detect it again as if it were brand new. 5cm comfortably
    covers normal depth-reading jitter on a stationary object while
    still telling apart two different objects placed near each other.
    """
    return [
        obj for obj in detected_objects
        if not any(_distance(obj["position"], done_pos) <= tolerance for done_pos in _done_positions)
    ]
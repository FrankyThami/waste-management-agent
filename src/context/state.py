"""
agent/state.py

Task state: everything the agent must not be trusted to remember on its
own. Kept as real data and changed only through the functions below
(never import the variables directly - a from-import copies a snapshot
that goes stale).

Three layers:
  - the CURRENT observation: each object is pending, done or skipped
  - what the arm is HOLDING right now
  - a SESSION registry: attempts per physical object, matched by position,
    so an object keeps its history even when a re-observation renames it
"""

import math

from src import policy

_objects = {}          # id -> {"waste_class", "position", "status"}  (current observation only)
_holding = None        # {"id", "waste_class", "position"} while the arm holds something
_done_positions = []   # where objects were successfully placed from
_registry = []         # [{"position", "attempts", "skipped", "approved"}] for the whole session
_place_attempts = 0    # failed placements of the object currently held
_abort_reason = None   # set once the task has been aborted
_mission = None        # the confirmed Mission for this run (None = no mission rules)

def new_session():
    """Forget everything. Call between separate runs inside one Python process."""
    global _holding, _place_attempts, _abort_reason
    _objects.clear()
    _done_positions.clear()
    _registry.clear()
    _holding = None
    _place_attempts = 0
    _abort_reason = None


def _distance(p1, p2):
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2 + (p1["z"] - p2["z"]) ** 2)


def _near(p1, p2):
    return _distance(p1, p2) <= policy.MATCH_TOLERANCE_M


def _record_for(position):
    """The registry entry for the physical object at this position (created if new)."""
    for rec in _registry:
        if _near(rec["position"], position):
            return rec
    rec = {"position": position, "attempts": 0, "skipped": False, "approved": False}
    _registry.append(rec)
    return rec


# ---------------------------------------------------------------- observation

def reset(detected_objects):
    """Called by observe_scene_tool with the objects it just saw (already filtered)."""
    _objects.clear()
    for obj in detected_objects:
        _objects[obj["id"]] = {
            "waste_class": obj["waste_class"],
            "position": obj["position"],
            "status": "pending",
        }


def filter_out_handled(detected_objects):
    """
    Drops objects that were already sorted or already skipped, matched by
    position. Needed because, without a real robot, a 'handled' object
    never leaves the camera's view and would be detected as brand new.

    Returns (remaining_objects, how_many_skipped_objects_were_seen_again).
    """
    remaining = []
    skipped_seen = 0
    for obj in detected_objects:
        pos = obj["position"]
        if _holding is not None and _near(pos, _holding["position"]):
            continue   # the arm is holding this object, so it is not on the table (the simulation leaves it in view)
        if any(_near(pos, done) for done in _done_positions):
            continue
        if any(rec["skipped"] and _near(pos, rec["position"]) for rec in _registry):
            skipped_seen += 1
            continue
        remaining.append(obj)
    return remaining, skipped_seen


def is_handled_position(position):
    """
    True if the agent itself picked an object up from this spot, or placed one that was there.
    The change report uses it to leave out changes the agent caused itself.
    """
    if _holding is not None and _near(position, _holding["position"]):
        return True
    return any(_near(position, done) for done in _done_positions)


# ---------------------------------------------------------------- abort

def aborted_message():
    """None normally; once the task is aborted, the refusal every tool should return."""
    if _abort_reason:
        return (f"rejected: the task was aborted ({_abort_reason}). "
                "No further actions are allowed - call safe_stop_tool.")
    return None


def abort(reason):
    global _abort_reason
    _abort_reason = reason


# ---------------------------------------------------------------- picking


# ---------------------------------------------------------------- mission

def set_mission(mission):
    """
    Sets the confirmed Mission for this run (or None for no mission rules).
    run_one_session() calls this at the start of EVERY run, so an old mission
    can never carry over into a later run. new_session() does not touch it.
    """
    global _mission
    _mission = mission


def get_mission():
    return _mission


def _in_mission(waste_class):
    """True if the current mission covers this class (always True when there is no mission)."""
    return _mission is None or waste_class in _mission.sort_order


def check_mission_allows_pick(obj_id):
    """
    None if the mission allows picking this object now, otherwise the reason it
    doesn't. Two rules, enforced here in code (not left to the model):
      1. the object's class must be part of the mission
      2. every class listed EARLIER in the mission's sort order must have
         nothing still pending
    """
    if _mission is None:
        return None
    waste_class = _objects[obj_id]["waste_class"]

    if waste_class not in _mission.sort_order:
        return (f"rejected: the mission does not include '{waste_class}' waste, so '{obj_id}' "
                f"must be left alone - do not pick it. {summary()}")

    for earlier in _mission.sort_order[: _mission.sort_order.index(waste_class)]:
        waiting = [oid for oid, info in _objects.items()
                   if info["status"] == "pending" and info["waste_class"] == earlier]
        if waiting:
            return (f"rejected: the mission says to sort '{earlier}' before '{waste_class}'. "
                    f"Still pending: {', '.join(waiting)}. Pick one of those first.")
    return None


def check_can_pick(obj_id):
    """None if picking this object is allowed right now, otherwise the reason it isn't."""
    blocked = aborted_message()
    if blocked:
        return blocked

    info = _objects[obj_id]
    if info["status"] == "done":
        return f"rejected: '{obj_id}' has already been placed - do not pick it again. {summary()}"
    if info["status"] == "skipped":
        return f"rejected: '{obj_id}' was skipped - do not try it again. {summary()}"
    if _holding is not None:
        return (f"rejected: the gripper is still holding '{_holding['id']}'. "
                "Place it with attempt_place_tool before picking anything else.")

    blocked = check_mission_allows_pick(obj_id)
    if blocked:
        return blocked

    rec = _record_for(info["position"])
    if rec["skipped"] or rec["attempts"] >= policy.MAX_PICK_ATTEMPTS:
        skip(obj_id)
        return f"rejected: '{obj_id}' has used up its attempts and is skipped - do not try it again. {summary()}"
    return None


def record_attempt(obj_id):
    """Counts one pick attempt against the physical object. Returns its attempt number."""
    rec = _record_for(_objects[obj_id]["position"])
    rec["attempts"] += 1
    return rec["attempts"]


def skip(obj_id):
    """Give up on this object for the rest of the session."""
    _record_for(_objects[obj_id]["position"])["skipped"] = True
    _objects[obj_id]["status"] = "skipped"


def finish_pick(obj_id, status):
    """
    Applies the outcome of a pick attempt. Returns a short note for the
    agent saying what happens next with this object ('' if nothing to add).
    """
    if status == "success":
        set_holding(obj_id)
        return ""

    rec = _record_for(_objects[obj_id]["position"])
    if status in ("infeasible", "unreachable"):
        skip(obj_id)
        return f"{obj_id} cannot be handled ({status}) and is now skipped - do not try it again."
    if rec["attempts"] >= policy.MAX_PICK_ATTEMPTS:
        skip(obj_id)
        return f"{obj_id} failed {rec['attempts']} times and is now skipped - do not try it again."
    left = policy.MAX_PICK_ATTEMPTS - rec["attempts"]
    return f"You may retry {obj_id} {left} more time(s)."


def set_holding(obj_id):
    """Remembers which object the arm is now holding."""
    global _holding
    info = _objects[obj_id]
    _holding = {"id": obj_id, "waste_class": info["waste_class"], "position": info["position"]}

def get_holding():
    """What the arm is holding right now: {'id', 'waste_class', 'position'}, or None."""
    return _holding
# ---------------------------------------------------------------- approval

def is_approved(obj_id):
    """True if a human already approved this physical object earlier this session."""
    return _record_for(_objects[obj_id]["position"])["approved"]


def mark_approved(obj_id):
    """Remembers that a human approved this physical object, so a later
    retry (e.g. after a failed grasp) doesn't ask again."""
    _record_for(_objects[obj_id]["position"])["approved"] = True


# ---------------------------------------------------------------- placing

def check_can_place():
    """None if placing is allowed right now, otherwise the reason it isn't."""
    blocked = aborted_message()
    if blocked:
        return blocked
    if _holding is None:
        return "rejected: the gripper is not holding anything - pick an object first."
    return None


def mark_holding_done():
    """Marks the held object as sorted and frees the gripper."""
    global _holding, _place_attempts
    if _holding is not None:
        info = _objects.get(_holding["id"])
        if info is not None and _near(info["position"], _holding["position"]):
            info["status"] = "done"
        _done_positions.append(_holding["position"])
    _holding = None
    _place_attempts = 0


def finish_place(status):
    """
    Applies the outcome of a place attempt.
    Returns (note_for_the_agent, must_abort_the_task).
    """
    global _place_attempts
    if status == "success":
        mark_holding_done()
        return "", False
    _place_attempts += 1
    if _place_attempts >= policy.MAX_PLACE_ATTEMPTS:
        return f"Placement failed {_place_attempts} times while holding an object.", True
    left = policy.MAX_PLACE_ATTEMPTS - _place_attempts
    return f"You may try placing again ({left} attempt(s) left).", False


# ---------------------------------------------------------------- reporting


def pending_ids():
    """Objects still waiting to be sorted. Objects the mission doesn't cover don't count."""
    return [obj_id for obj_id, info in _objects.items()
            if info["status"] == "pending" and _in_mission(info["waste_class"])]


def summary():
    """A short line for the agent to see what's left, in plain language."""
    pending = pending_ids()
    skipped = [obj_id for obj_id, info in _objects.items() if info["status"] == "skipped"]
    left_alone = [obj_id for obj_id, info in _objects.items()
                  if info["status"] == "pending" and not _in_mission(info["waste_class"])]
    if pending:
        text = f"{len(pending)} object(s) still pending: " + ", ".join(
            f"{obj_id} ({_objects[obj_id]['waste_class']})" for obj_id in pending
        )
    else:
        text = "No pending objects remain in this observation."
    if skipped:
        text += " Skipped (given up on): " + ", ".join(
            f"{obj_id} ({_objects[obj_id]['waste_class']})" for obj_id in skipped
        )
    if left_alone:
        text += " Left alone (not part of the mission - do not pick): " + ", ".join(
            f"{obj_id} ({_objects[obj_id]['waste_class']})" for obj_id in left_alone
        )
    return text







"""
agent/langchain_tools.py

Wraps the 4 pure functions in agent/tools.py as LangChain @tool
functions, so Gemini can call them through create_agent.

tools.py itself stays framework-agnostic (no LangChain import there) -
this file is the only place that knows LangChain exists. It is also where
the recovery rules from agent/state.py are enforced: the model can ask
for anything, but the code decides what is actually allowed.
"""

import time

from langchain_core.tools import tool

from src import policy
from src.context import state
from src.context.change_report import describe_changes
from src.reliability import guardrails, run_stats, stale_guard
from src.actions.tools import observe_scene, attempt_pick, attempt_place, safe_stop

_last_observation = {}   # object id -> object, from the most recent observation
_last_bins = []          # the bins seen in the most recent observation


def _guarded(check_function, believed):
    """
    Takes a fresh look (not remembered as the 'previous look'), runs the guard check on ONE target
    and records how long it took. Returns (check, abort_text); abort_text is None unless too many
    re-checks in a row refused, in which case the task has been aborted and the arm halted.
    """
    started = time.perf_counter()
    check = check_function(believed, observe_scene(remember=False))
    run_stats.record_guard(check["ok"], time.perf_counter() - started)

    in_a_row = run_stats.consecutive_refusals()
    if check["ok"] or in_a_row < policy.MAX_CONSECUTIVE_GUARD_REFUSALS:
        return check, None

    reason = f"The scene kept changing: {in_a_row} safety re-checks in a row refused to move the arm"
    state.abort(reason)
    safe_stop(reason)
    return check, (f"{check['message']}\nThe arm has been halted and the task is ABORTED ({reason}). "
                   "Call safe_stop_tool to finish.")


@tool
def observe_scene_tool() -> str:
    """Look at the table through the camera and report what is visible right now.

    Returns every waste object (id, waste class, position, size) and every bin
    (colour, position of its rim, size). Positions are in metres in the
    robot's base frame; z is height above the table.
    After the first look it also reports what has changed since your previous
    look: objects or bins that moved, disappeared or appeared. Ids are
    renumbered on every look.
    """
    blocked = state.aborted_message()
    if blocked:
        return blocked

    observation = observe_scene()
    if observation.get("error"):
        return (f"Observation failed: {observation['error']}\n"
                "Nothing is known about the scene right now.")

    objects, skipped_seen = state.filter_out_handled(observation["objects"])
    bins = observation["bins"]
    _last_observation.clear()
    _last_observation.update({obj["id"]: obj for obj in objects})
    _last_bins.clear()
    _last_bins.extend(bins)
    state.reset(objects)

    lines = []
    if objects:
        lines.append(f"Detected {len(objects)} object(s):")
        lines += [
            f"- {obj['id']}: {obj['waste_class']} at position {obj['position']}, "
            f"about {obj['width_m'] * 100:.0f} x {obj['height_m'] * 100:.0f} cm"
            for obj in objects
        ]
    else:
        lines.append("No objects detected in the scene.")

    if bins:
        lines.append(f"Detected {len(bins)} bin(s):")
        lines += [
            f"- {b['colour']} bin: rim at position {b['position']}, "
            f"about {b['width_m'] * 100:.0f} x {b['height_m'] * 100:.0f} cm"
            for b in bins
        ]
    else:
        lines.append("No bins detected in the scene.")

    if observation["unmeasured"]:
        lines.append("Seen but could not be measured (no reliable depth, so they cannot be used): "
                     + ", ".join(observation["unmeasured"]))
    if skipped_seen:
        lines.append(f"(Ignoring {skipped_seen} previously skipped object(s) still in view "
                     "- do not try them again.)")

    holding = state.get_holding()
    if holding:
        lines.append(f"You are currently holding a {holding['waste_class']} object "
                     "(picked up earlier, so it is not on the table).")
    lines += describe_changes(observation.get("changes"), state.is_handled_position)
    return "\n".join(lines)


@tool
def attempt_pick_tool(target_id: str) -> str:
    """Attempt to pick up one previously observed object, by its id.

    Args:
        target_id: The id of the object to pick, exactly as it appeared
            in observe_scene's result (e.g. "obj_0").
    """
    target = _last_observation.get(target_id)
    if target is None:
        return f"rejected: no object with id '{target_id}' - call observe_scene_tool first, or check the id."

    refusal = state.check_can_pick(target_id)
    if refusal:
        return refusal

    # Stage 3.2/3.4: before the human is asked or the arm moves, make sure the object is still where it was seen.
    print(f"[guard] Re-checking the {target['waste_class']} object {target_id} before the arm moves...")
    check, abort_text = _guarded(stale_guard.check_object, target)
    if abort_text:
        return abort_text
    if not check["ok"]:
        return check["message"]
    target = check["target"]   # the fresh measurement, same id

    reason = guardrails.approval_reason(target)
    if reason and not state.is_approved(target_id):
        if guardrails.ask_human_approval(target, reason):
            state.mark_approved(target_id)
        else:
            state.skip(target_id)
            return (f"declined: a human declined to approve picking this {target['waste_class']} object. "
                     f"It is now skipped - do not try it again. {state.summary()}")

    state.record_attempt(target_id)
    result = attempt_pick(target)
    note = state.finish_pick(target_id, result["status"])

    text = f"{result['status']}: {result['reason']}"
    return f"{text}\n{note}" if note else text


@tool
def attempt_place_tool(bin_colour: str) -> str:
    """Release the object you are currently holding into one of the bins you have seen.

    Only possible while holding an object (after a successful attempt_pick_tool).

    Args:
        bin_colour: The colour of the bin to release into, written as the colour
            appears in the latest observe_scene_tool result (for example "red").
    """
    refusal = state.check_can_place()
    if refusal:
        return refusal

    wanted = bin_colour.strip().lower()
    matches = [b for b in _last_bins if b["colour"].strip().lower() == wanted]
    if not matches:
        seen = ", ".join(b["colour"] for b in _last_bins) or "none"
        return (f"rejected: no bin with the colour '{bin_colour}' was seen in the latest observation "
                f"(bins seen: {seen}). Nothing was released - you are still holding the object.")
    if len(matches) > 1:
        return (f"rejected: {len(matches)} bins with the colour '{bin_colour}' were seen, so it is unclear "
                "which one to use. Nothing was released - you are still holding the object.")

    # Stage 3.2/3.4: make sure this bin is still where it was seen before the arm moves over it.
    print(f"[guard] Re-checking the {matches[0]['colour']} bin before the arm moves...")
    check, abort_text = _guarded(stale_guard.check_bin, matches[0])
    if abort_text:
        return abort_text
    if not check["ok"]:
        return f"{check['message']} Nothing was released - you are still holding the object."
    bin_info = check["target"]   # the fresh measurement

    holding = state.get_holding()
    result = attempt_place(bin_info, holding["waste_class"])
    note, must_abort = state.finish_place(result["status"])
    text = f"{result['status']}: {result['reason']}"

    if must_abort:
        reason = "Placement failed repeatedly while holding an object"
        state.abort(reason)
        safe_stop(reason)
        return (f"{text}\n{note}\n"
                "The arm has been halted and the task is ABORTED. Call safe_stop_tool to finish.")
    if note:
        return f"{text}\n{note}"
    return f"{text}\n\n{state.summary()}"


@tool
def safe_stop_tool(reason: str) -> str:
    """Immediately halt the arm and end the task. Call this if something
    looks wrong, an object can't be handled safely, or the task is complete.

    Args:
        reason: A short explanation of why the task is stopping.
    """
    result = safe_stop(reason)
    print(run_stats.report_and_reset())
    return f"{result['status']}: {result['reason']}"
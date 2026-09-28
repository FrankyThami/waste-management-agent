"""
agent/langchain_tools.py

Wraps the 4 pure functions in agent/tools.py as LangChain @tool
functions, so Gemini can call them through create_agent.

tools.py itself stays framework-agnostic (no LangChain import there) -
this file is the only place that knows LangChain exists.
"""

from langchain_core.tools import tool

from agent.tools import observe_scene, attempt_pick, attempt_place, safe_stop
from agent import state

# Holds the objects from the most recent observe_scene() call, keyed by
# id, so attempt_pick can look one up by a short id instead of asking
# the model to retype an exact 3D position.
_last_observation = {}


@tool
def observe_scene_tool() -> str:
    """Capture the current scene from the camera and list all detected waste objects.

    Call this first, and again any time you need an updated view of the
    workspace (e.g. after placing an object, to see what's left).
    """
    objects = observe_scene()
    objects = state.filter_out_completed(objects)
    _last_observation.clear()
    _last_observation.update({obj["id"]: obj for obj in objects})
    state.reset(objects)

    if not objects:
        return "No objects detected in the scene."

    lines = [
        f"- {obj['id']}: {obj['waste_class']} at position {obj['position']}"
        for obj in objects
    ]
    return f"Detected {len(objects)} object(s):\n" + "\n".join(lines)


@tool
def attempt_pick_tool(target_id: str) -> str:
    """Attempt to pick up one previously observed object, by its id.

    Args:
        target_id: The id of the object to pick, exactly as it appeared
            in observe_scene's result (e.g. "obj_0").
    """
    target = _last_observation.get(target_id)
    if target is None:
        return f"No object with id '{target_id}' - call observe_scene first, or check the id."

    if not state.is_pending(target_id):
        return f"'{target_id}' has already been placed - do not pick it again. {state.summary()}"

    state.record_attempt(target_id)
    result = attempt_pick(target)
    if result["status"] == "success":
        state.set_holding(target_id)
    return f"{result['status']}: {result['reason']}"


@tool
def attempt_place_tool(waste_class: str) -> str:
    """Place the currently held object into the bin for its waste class.

    Only call this immediately after a successful attempt_pick_tool.

    Args:
        waste_class: One of paper, metal, organic, unsorted, plastic, glass.
    """
    result = attempt_place(waste_class)
    if result["status"] == "success":
        state.mark_holding_done()
    return f"{result['status']}: {result['reason']}\n\n{state.summary()}"


@tool
def safe_stop_tool(reason: str) -> str:
    """Immediately halt the arm and end the task. Call this if something
    looks wrong, an object can't be handled safely, or the task is complete.

    Args:
        reason: A short explanation of why the task is stopping.
    """
    result = safe_stop(reason)
    return f"{result['status']}: {result['reason']}"
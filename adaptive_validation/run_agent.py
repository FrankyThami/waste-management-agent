"""
adaptive_validation/run_agent.py

Day 3: full 8-tool autonomous agent. Pauses once, right after the
agent first locates the red dustbin, so you can physically move it -
watch the live camera window to confirm the move actually registers,
then press Enter and see whether the agent notices and recovers.

Run from the project root, venv active:
    python -m adaptive_validation.run_agent
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from adaptive_validation import scene_state
from adaptive_validation.agent_loop import run_tool_loop
from adaptive_validation.tools import observe_scene, reobserve_scene, locate_object
from adaptive_validation.action_tools import (
    check_pick_feasibility, check_reachability,
    pick_object, verify_pick, place_in_bin, verify_placement,
)

load_dotenv()
MODEL_NAME = "gemini-robotics-er-2-preview"

TOOLS = [
    observe_scene, reobserve_scene, locate_object,
    check_pick_feasibility, check_reachability,
    pick_object, verify_pick, place_in_bin, verify_placement,
]
TOOL_MAP = {t.name: t for t in TOOLS}
INSTRUCTION = "Pick the yellow paper and put it in the red dustbin."

_has_paused = False


def dustbin_just_located(tool_name: str, tool_args: dict) -> bool:
    global _has_paused
    if _has_paused:
        return False
    if tool_name == "locate_object" and tool_args.get("target_description") == "red dustbin":
        _has_paused = True
        return True
    return False


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key).bind_tools(TOOLS)

    scene_state.start_camera()
    try:
        run_tool_loop(
            llm, TOOL_MAP, [HumanMessage(content=INSTRUCTION)],
            max_steps=30,
            pause_condition=dustbin_just_located,
        )
    finally:
        scene_state.stop_camera()


if __name__ == "__main__":
    main()
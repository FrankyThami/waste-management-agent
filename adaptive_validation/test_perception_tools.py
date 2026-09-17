"""
adaptive_validation/test_perception_tools.py

Day 2, Step 2: let Gemini ER2 drive observe_scene() and locate_object()
itself. See adaptive_validation/agent_loop.py for the loop logic.

Run from the project root, venv active:
    python -m adaptive_validation.test_perception_tools
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from adaptive_validation import scene_state
from adaptive_validation.agent_loop import run_tool_loop
from adaptive_validation.tools import observe_scene, locate_object

load_dotenv()
MODEL_NAME = "gemini-robotics-er-2-preview"
TOOLS = [observe_scene, locate_object]
TOOL_MAP = {t.name: t for t in TOOLS}


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key).bind_tools(TOOLS)

    scene_state.start_camera()
    try:
        messages = [HumanMessage(content=(
            "Find the yellow paper and the red dustbin and tell me both of "
            "their positions. Observe the scene first if you haven't already."
        ))]
        run_tool_loop(llm, TOOL_MAP, messages)
    finally:
        scene_state.stop_camera()


if __name__ == "__main__":
    main()
"""
agent/run_agent.py

Day 2, Step 5: the actual autonomous agent. Combines the model, the 4
tools, and the system prompt into one create_agent() loop.

Run it, give it the instruction, and it decides everything else itself.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.langchain_tools import (
    observe_scene_tool, attempt_pick_tool, attempt_place_tool, safe_stop_tool,
)
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.utils import get_response_text

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-robotics-er-2-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

agent = create_agent(
    model=model,
    tools=[observe_scene_tool, attempt_pick_tool, attempt_place_tool, safe_stop_tool],
    system_prompt=AGENT_SYSTEM_PROMPT,
)


def print_trace(messages):
    """Prints every decision and result in the run, in order - useful for
    watching what the agent actually chose to do, not just the final line."""
    for msg in messages:
        kind = msg.__class__.__name__
        if kind == "HumanMessage":
            print(f"[Instruction] {get_response_text(msg)}")
        elif kind == "AIMessage":
            text = get_response_text(msg)
            if text:
                print(f"[Agent] {text}")
            for call in getattr(msg, "tool_calls", None) or []:
                print(f"[Agent -> calls] {call['name']}({call['args']})")
        elif kind == "ToolMessage":
            print(f"[Tool result] {msg.content}")
        print()


def run_one_session(instruction="Sort all the visible waste into the correct bins."):
    return agent.invoke(
        {"messages": [{"role": "user", "content": instruction}]},
        config={"recursion_limit": 60},
    )


if __name__ == "__main__":
    print("=== Autonomous waste-sorting agent ===\n")
    result = run_one_session()
    print_trace(result["messages"])
"""
src/interface/run_agent.py

The autonomous agent's entry point. The operator types a task in plain language;
it goes straight to the agent (Gemini), together with its 4 tools and a short
factual system prompt. The agent decides what to do and in what order.

The code only enforces safety (guardrails.py, state.py, the MATLAB limits) and
records what happened (task_logger.py) - it never decides the task.

Run from the project root:
    python -m src.interface.run_agent
        opens the home screen: type a task, I = instructions, Q = quit
    python -m src.interface.run_agent "Put the organic waste into the green bin"
        runs that one task straight away, then exits
"""

import os
import sys
import traceback

# Lets this file run both as a module and as a plain script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src import policy
from src.actions import hal, langchain_tools, observation
from src.actions.langchain_tools import (
    observe_scene_tool, attempt_pick_tool, attempt_place_tool, safe_stop_tool,
)
from src.context import state
from src.reasoning.prompts import AGENT_SYSTEM_PROMPT
from src.reliability import run_stats, task_logger
from src.reliability.middleware import logging_middleware, model_call_counter
from src.utils import get_response_text

load_dotenv()

TITLE = "=== [ Autonomous Waste-Sorting Agent ] ==="
HOME_HINT = "Type your task and press Enter to continue   |   I: instructions   |   Q: quit"

# The six waste classes of the project.
WASTE_CLASSES = "paper, metal, organic, unsorted waste, plastic, glass"

# What the instructions box shows (keep each line under about 66 characters).
INSTRUCTIONS = [
    "WASTE CLASSES THE ROBOT CAN SORT",
    f"  {WASTE_CLASSES}",
    "",
    "HOW TO GIVE A TASK",
    "  Write it in plain language: which waste goes into which bin",
    "  colour, and in what order if the order matters.",
    "  Example: Put the organic waste into the green bin, then the",
    "  plastic into the blue bin.",
    "  Name only bin colours that are really on the table.",
    "",
    "WHAT THE ROBOT WILL NOT DO",
    "  - Anything other than sorting waste into bins.",
    "  - Touch people, hands or objects that are not waste.",
    "  - Pick items that are too large or too risky for the gripper:",
    "    these are refused or need a human's approval.",
    "  - Switch off its safety checks: no command can override them.",
    "",
    "PLEASE",
    "  - Do not send unauthorised or unsafe tasks.",
    "  - Keep hands and objects away from the arm while it moves.",
    "  - To stop a running task, press Ctrl+C.",
]


# ---------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------

def build_agent():
    """The agent: Gemini + 4 tools + a short factual system prompt."""
    model = ChatGoogleGenerativeAI(
        model="gemini-robotics-er-2-preview",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    return create_agent(
        model=model,
        tools=[observe_scene_tool, attempt_pick_tool, attempt_place_tool, safe_stop_tool],
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=[logging_middleware, model_call_counter],
    )


def print_trace(messages):
    """Prints every decision and result in the run, in order - useful for
    watching what the agent actually chose to do, not just the final line."""
    for msg in messages:
        kind = msg.__class__.__name__
        if kind == "HumanMessage":
            print(f"[Operator] {get_response_text(msg)}")
        elif kind == "AIMessage":
            text = get_response_text(msg)
            if text:
                print(f"[Agent] {text}")
            for call in getattr(msg, "tool_calls", None) or []:
                print(f"[Agent -> calls] {call['name']}({call['args']})")
        elif kind == "ToolMessage":
            print(f"[Tool result] {msg.content}")
        print()


def print_summary():
    """Shows the numbers this run just recorded to session_summary.csv."""
    s = task_logger.get_last_summary()
    if s is None:
        return
    print("=== Session summary ===")
    print(f"Completion time : {s['completion_time_s']} s")
    print(f"Tool calls      : {s['tool_calls']}")
    print(f"Model calls     : {s['model_calls']}")
    print(f"Re-observations : {s['re_observations']}")
    print(f"Picks           : {s['picks_succeeded']}/{s['pick_attempts']} succeeded")
    print(f"Placements      : {s['placements_succeeded']}/{s['place_attempts']} succeeded")
    print(f"Stop reason     : {s['stop_reason'] or '(agent did not call safe_stop)'}")


def run_one_session(command):
    """Runs the agent once. The operator's sentence goes to the agent exactly as typed."""
    task_logger.start_session(command)
    agent = build_agent()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": command}]},
            config={"recursion_limit": policy.RECURSION_LIMIT},
        )
    except (Exception, KeyboardInterrupt) as e:   # Ctrl+C is logged too, then passed on
        task_logger.end_session("error", f"{type(e).__name__}: {e}")
        raise
    task_logger.end_session("finished", "Agent loop ended normally")
    return result


def run_task(command):
    """One full task: run the agent, then print what it did and the numbers."""
    print(f'\nOperator: "{command}"\nStarting...\n')
    result = run_one_session(command)
    print_trace(result["messages"])
    print_summary()


# ---------------------------------------------------------------------
# Between tasks
# ---------------------------------------------------------------------

def start_fresh_task():
    """
    Forgets everything the agent remembered from the previous task, so a new task
    starts exactly like a new run of the program.
    """
    state.new_session()                          # handled / skipped objects, what is held, abort flag
    observation.reset_look_memory()              # the "previous look" used for change reports
    run_stats.report_and_reset()                 # zero the look counters (the returned text is not needed)
    langchain_tools._last_observation.clear()    # object ids from the last task's last look
    langchain_tools._last_bins.clear()


def halt_arm():
    """Tells the robot backend to stop all motion (used when the operator presses Ctrl+C)."""
    try:
        hal.simulate_stop()
    except Exception as error:
        print(f"Could not send the stop command: {error}")


def print_run_error(error):
    """Quota errors get a plain explanation; anything else prints in full so it can be debugged."""
    print(f"\nThe task stopped because of an error ({type(error).__name__}).")
    if "RESOURCE_EXHAUSTED" in str(error):
        print("Gemini's request quota is used up. Check your usage at "
              "https://ai.dev/rate-limit and try again later.")
    else:
        traceback.print_exc()


# ---------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------

def backend_name():
    return "MATLAB simulation" if hal.HAL_MODE == "matlab" else "print-only simulation (no MATLAB)"


def print_header():
    print(f"\n{TITLE}\n")
    print(f"Robot backend: {backend_name()}\n")


def print_box(title, lines):
    """Prints lines of text inside a simple box."""
    width = max(len(title), max(len(line) for line in lines))
    border = "+" + "-" * (width + 2) + "+"
    print(border)
    print(f"| {title.center(width)} |")
    print(border)
    for line in lines:
        print(f"| {line.ljust(width)} |")
    print(border)


def show_instructions():
    """The instructions screen. Returns when the operator types B."""
    print()
    print_box("INSTRUCTIONS", INSTRUCTIONS)
    while True:
        try:
            key = input("\nPress B to go back to the main home > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if key in ("b", "back"):
            return
        print("Type B and press Enter to go back.")


def home_screen():
    """
    The main home. Returns the task the operator typed, or None if they want to quit.
    The single letters I, B and Q are commands, so a task cannot be just one of those letters.
    """
    print_header()
    while True:
        print(HOME_HINT)
        try:
            text = input("How can I help you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        key = text.lower()
        if key in ("q", "quit", "exit"):
            return None
        if key in ("i", "instructions"):
            show_instructions()
            print_header()
            continue
        if not text:
            print("Please type a task first (I for instructions, Q to quit).\n")
            continue
        return text


def main():
    # A sentence on the command line: run that one task straight away, as before.
    command = " ".join(sys.argv[1:]).strip()
    if command:
        print_header()
        run_task(command)
        return

    while True:
        command = home_screen()
        if command is None:
            print("\nGoodbye. Have a nice Day Sir")
            return

        start_fresh_task()
        try:
            run_task(command)
        except KeyboardInterrupt:
            print("\n\nStopped by the operator (Ctrl+C). Telling the arm to halt...")
            halt_arm()
        except Exception as error:
            print_run_error(error)
        print("\n" + "-" * 60)
        print("Task finished. Back to the main home.")


if __name__ == "__main__":
    main()
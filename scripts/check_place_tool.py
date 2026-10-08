"""
scripts/check_place_tool.py - Stage 2.5b live check.

You play the agent for ONE object. The script looks at the real table with the real
observe_scene tool, then you choose an object id and a bin colour, and the real
attempt_pick and attempt_place tools run exactly as they will for the agent.

Run from the project root:   python scripts/check_place_tool.py
"""

import os
import sys

# Let this script import from src/ no matter where it is started from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.actions.langchain_tools import (   # noqa: E402
    observe_scene_tool, attempt_pick_tool, attempt_place_tool,
)


def main():
    print("Looking at the real table with the agent's observe_scene tool...\n")
    print("\n=== What the agent would see ===")
    print(observe_scene_tool.invoke({}))

    object_id = input("\nWhich object id should be picked (for example obj_0)? ").strip()
    pick_text = attempt_pick_tool.invoke({"target_id": object_id})
    print(f"\n[pick] {pick_text}")
    if not pick_text.startswith("success"):
        print("\nThe pick did not succeed, so there is nothing to place. Stopping.")
        return

    while True:
        colour = input("\nWhich bin colour should it be released into? (q = stop here) ").strip()
        if colour.lower() in ("q", "quit"):
            print("Stopped while still holding the object.")
            return
        place_text = attempt_place_tool.invoke({"bin_colour": colour})
        print(f"\n[place] {place_text}")
        if place_text.startswith("success"):
            return


if __name__ == "__main__":
    main()
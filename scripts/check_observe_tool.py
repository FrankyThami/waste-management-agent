"""
scripts/check_observe_tool.py - Stage 2.5a live check.

Calls the agent's real observe_scene tool on the real table and prints exactly
the text the agent receives, then opens the annotated image of that observation.

Run from the project root:   python scripts/check_observe_tool.py
"""

import glob
import os
import sys

# Let this script import from src/ no matter where it is started from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.actions.langchain_tools import observe_scene_tool   # noqa: E402


def main():
    print("Calling the agent's observe_scene tool on the real table...\n")
    text = observe_scene_tool.invoke({})
    print("\n=== What the agent receives ===")
    print(text)

    images = sorted(glob.glob(os.path.join(PROJECT_ROOT, "logs", "observations", "*_annotated.png")))
    if images:
        try:
            os.startfile(images[-1])  # Windows: opens the newest annotated image
        except (AttributeError, OSError):
            print(f"\n(Open this image to check it: {images[-1]})")


if __name__ == "__main__":
    main()
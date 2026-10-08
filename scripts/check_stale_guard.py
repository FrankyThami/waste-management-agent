"""
scripts/check_stale_guard.py - live check of the pre-move guard (Stage 3.2).

The agent believes in the positions from ONE look. The guard re-checks them with a fresh look.
Round 1: nothing changed -> the guard must say OK.
Round 2: you move the item and the bin -> the guard must refuse.

Run:  python scripts/check_stale_guard.py
"""

import os
import sys

# Let this script import from src/ no matter where it is started from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.actions.observation import observe                      # noqa: E402
from src.reliability.stale_guard import check_bin, check_object   # noqa: E402


def run_checks(round_name, believed_object, believed_bin):
    fresh = observe(remember=False)   # a re-check look, exactly like the guard does in the agent
    print(f"\n--- {round_name} ---")
    for result in (check_object(believed_object, fresh), check_bin(believed_bin, fresh)):
        if result["ok"]:
            p = result["target"]["position"]
            print(f"    -> the arm would be allowed to move (fresh position x={p['x']:.3f}, y={p['y']:.3f})")
        else:
            print(f"    -> the arm would NOT move. The agent would receive:\n       {result['message']}")


def main():
    print("LOOK 1 - this is what the agent 'believes'. Leave the table as it is...")
    first = observe()
    if first["error"] or not first["objects"] or not first["bins"]:
        print("I need at least one measured object and one bin on the table. Check the scene and try again.")
        return

    believed_object, believed_bin = first["objects"][0], first["bins"][0]
    print(f"Guarding the {believed_object['waste_class']} object {believed_object['id']} "
          f"and the {believed_bin['colour']} bin.")

    run_checks("ROUND 1: nothing moved", believed_object, believed_bin)

    print(f"\nNow move the {believed_object['waste_class']} object AND the {believed_bin['colour']} bin "
          "by about 10 cm.")
    input("Press Enter when you have moved them...")

    run_checks("ROUND 2: both moved", believed_object, believed_bin)


if __name__ == "__main__":
    main()
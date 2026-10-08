"""
scripts/check_scene_changes.py - live check of change detection (Stage 3.1).

Look at the table -> you move things by hand -> look again -> print what changed.
It uses the same observe() the agent uses, so this is exactly what the agent will receive.

Run:  python scripts/check_scene_changes.py
"""

import os
import sys

# Let this script import from src/ no matter where it is started from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ADJUST this line if your observation.py is not in src/actions/
from src.actions.observation import observe  # noqa: E402

LABEL_KEYS = {"objects": "waste_class", "bins": "colour"}


def describe(status, entry, label_key):
    """One readable line for one entry of the change report."""
    label = entry.get(label_key, "")
    if status in ("unchanged", "moved"):
        return (f"{entry['was_id']} -> {entry['id']}  {label:<10} "
                f"shifted {entry['shift_m'] * 1000:.0f} mm")
    if status == "reclassified":
        return (f"{entry['was_id']} -> {entry['id']}  was '{entry['was']}', "
                f"now '{entry['now']}' (same spot)")
    if status == "missing":
        return f"{entry['was_id']}  {label:<10} no longer seen"
    return f"{entry['id']}  {label:<10} newly seen"


def print_report(changes):
    print(f"\nChanges since the previous look ({changes['seconds_since_previous_look']} s earlier):")
    for kind, label_key in LABEL_KEYS.items():
        print(f"\n{kind.upper()}")
        group = changes[kind]
        if not any(group.values()):
            print("  (nothing seen)")
        for status in ("moved", "reclassified", "missing", "new", "unchanged"):
            for entry in group[status]:
                print(f"  {status.upper():<12} {describe(status, entry, label_key)}")


def main():
    print("LOOK 1 - leave the table exactly as it is...")
    first = observe()
    if first["error"]:
        print(f"Look 1 failed: {first['error']}")
        return
    print(f"Look 1 saw {len(first['objects'])} object(s) and {len(first['bins'])} bin(s).")

    print("\nNow change the table by hand: move ONE item and ONE bin by about 5 cm. "
          "Leave everything else alone.")
    input("Press Enter when you have moved them...")

    print("\nLOOK 2...")
    second = observe()
    if second["error"]:
        print(f"Look 2 failed: {second['error']}")
        return
    print(f"Look 2 saw {len(second['objects'])} object(s) and {len(second['bins'])} bin(s).")
    print_report(second["changes"])


if __name__ == "__main__":
    main()
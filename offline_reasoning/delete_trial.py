"""
offline_reasoning/delete_trial.py

Standalone utility to delete ANY past trial by number - not just the one
you just recorded. Also deletes the matching saved image if it exists.

Usage (from the project root, venv active):
    python -m offline_reasoning.delete_trial plastic separated 7
"""

import sys
from pathlib import Path

from offline_reasoning.results_store import delete_trial

IMAGES_ROOT = Path("offline_reasoning/images")


def main():
    if len(sys.argv) != 4:
        print("Usage: python -m offline_reasoning.delete_trial <class> <environment> <trial_number>")
        sys.exit(1)

    waste_class, environment, trial_number = sys.argv[1], sys.argv[2], int(sys.argv[3])

    deleted = delete_trial(waste_class, environment, trial_number)
    if not deleted:
        print(f"No trial {trial_number} found for {waste_class} / {environment}.")
        return

    image_path = IMAGES_ROOT / environment / waste_class / f"test{trial_number}.jpg"
    image_path.unlink(missing_ok=True)

    print(f"Deleted trial {trial_number} for {waste_class} / {environment} "
          f"(and its saved image, if it existed).")


if __name__ == "__main__":
    main()
"""
offline_reasoning/results_store.py

Reads/writes the per-trial results tables for Offline Reasoning
Validation - one CSV per (waste class, environment) pair, in the exact
column layout from the "Process of OFFLINE Reasoning" spec:

    Trial | Ground Truth | Predicted | Paper | Metal | Organic | Unsorted | Plastic | Glass

Each row is one physical item shown to the camera. The one-hot columns
mark which single class Gemini predicted for that trial.
"""

import csv
from pathlib import Path

RESULTS_ROOT = Path("offline_reasoning/results")

# Canonical display labels, matching the reference table exactly.
# Handles the fact Gemini returns "unsorted waste" (from our prompt's
# category list) but the table header just says "Unsorted".
CLASS_LABELS = {
    "paper": "Paper",
    "metal": "Metal",
    "organic": "Organic",
    "unsorted waste": "Unsorted",
    "unsorted": "Unsorted",
    "plastic": "Plastic",
    "glass": "Glass",
}
COLUMN_ORDER = ["Paper", "Metal", "Organic", "Unsorted", "Plastic", "Glass"]
FIELDNAMES = ["Trial", "Ground Truth", "Predicted"] + COLUMN_ORDER


def normalize_class(raw_class: str) -> str:
    """Turns "unsorted waste" / "PLASTIC" / etc. into the exact display
    label used in the table ("Unsorted" / "Plastic")."""
    key = raw_class.strip().lower()
    if key not in CLASS_LABELS:
        raise ValueError(
            f"Unknown waste class '{raw_class}'. Expected one of: paper, metal, organic, unsorted waste, plastic, glass"
        )
    return CLASS_LABELS[key]


def csv_path(waste_class: str, environment: str) -> Path:
    """e.g. offline_reasoning/results/separated/plastic.csv"""
    folder = RESULTS_ROOT / environment.lower()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{waste_class.lower()}.csv"


def read_trials(waste_class: str, environment: str) -> list[dict]:
    path = csv_path(waste_class, environment)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(waste_class: str, environment: str, rows: list[dict]) -> None:
    path = csv_path(waste_class, environment)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def next_trial_number(waste_class: str, environment: str) -> int:
    rows = read_trials(waste_class, environment)
    if not rows:
        return 1
    return max(int(r["Trial"]) for r in rows) + 1


def append_trial(waste_class: str, environment: str, ground_truth: str, predicted: str) -> int:
    """
    Adds one trial row. Returns the trial number assigned, so the caller
    can name the matching saved image file (e.g. test{trial_number}.jpg).
    """
    ground_truth = normalize_class(ground_truth)
    predicted = normalize_class(predicted)

    trial_number = next_trial_number(waste_class, environment)
    row = {"Trial": trial_number, "Ground Truth": ground_truth, "Predicted": predicted}
    for col in COLUMN_ORDER:
        row[col] = 1 if col == predicted else 0

    rows = read_trials(waste_class, environment)
    rows.append(row)
    _write_all(waste_class, environment, rows)
    return trial_number


def delete_trial(waste_class: str, environment: str, trial_number: int) -> bool:
    """
    Removes one trial row by its Trial number. Does NOT renumber the
    remaining trials or touch any saved image file - only edits the CSV.
    Returns True if a row was found and deleted, False otherwise.
    """
    rows = read_trials(waste_class, environment)
    kept = [r for r in rows if int(r["Trial"]) != trial_number]
    if len(kept) == len(rows):
        return False
    _write_all(waste_class, environment, kept)
    return True
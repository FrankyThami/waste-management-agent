"""
Stage 3 deliverable script.

Runs Gemini Robotics ER 2 (using Prompt 1) against every scene in
labelled_scenes/, compares its category guesses to the ground-truth
labels, and prints a basic accuracy summary.

NOTE - simplification: this compares the overall *list* of predicted
categories against the ground-truth categories per scene (a multiset
match), not a full confusion matrix with per-object correspondence.
The full Benchmark 1 confusion matrix (Predicted vs Actual per class,
with precision/recall/F1) is built in Stage 8.
"""

import os
import json
import logging
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from prompts.prompts_v1 import PROMPT_1_OBJECT_CENTERS

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("accuracy_test")

# ---------- Gemini setup ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-robotics-er-2-preview"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
def call_gemini(contents):
    return client.models.generate_content(model=MODEL_NAME, contents=contents)


def parse_json_response(text):
    """Gemini is told to return raw JSON, but strip markdown fences defensively."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned)


def score_scene(predicted_categories, actual_categories):
    """
    Basic multiset comparison: for each category, how many predicted
    items could be matched to an actual item of the same category.
    Returns (correct_count, total_actual_count).
    """
    predicted_counts = Counter(predicted_categories)
    actual_counts = Counter(actual_categories)

    correct = 0
    for category, actual_count in actual_counts.items():
        correct += min(actual_count, predicted_counts.get(category, 0))

    return correct, len(actual_categories)


def main():
    with open("labelled_scenes/labels.json") as f:
        dataset = json.load(f)

    total_correct = 0
    total_objects = 0

    for scene in dataset["scenes"]:
        scene_id = scene["scene_id"]
        image_path = Path("labelled_scenes") / scene["image"]

        logger.info(f"Testing {scene_id} ({image_path})...")
        image = Image.open(image_path)

        response = call_gemini([PROMPT_1_OBJECT_CENTERS, image])

        try:
            predictions = parse_json_response(response.text)
        except json.JSONDecodeError:
            print(f"\n{scene_id}: could not parse model response as JSON:")
            print(response.text)
            continue

        predicted_categories = [p["material"] for p in predictions]
        actual_categories = [o["category"] for o in scene["objects"]]

        correct, total = score_scene(predicted_categories, actual_categories)
        total_correct += correct
        total_objects += total

        print(f"\n--- {scene_id} ---")
        print(f"Ground truth categories : {actual_categories}")
        print(f"Predicted categories    : {predicted_categories}")
        print(f"Correct: {correct}/{total}")

    print("\n===== OVERALL RESULTS =====")
    if total_objects > 0:
        accuracy = total_correct / total_objects * 100
        print(f"Total correct: {total_correct}/{total_objects}")
        print(f"Overall accuracy: {accuracy:.1f}%")
    else:
        print("No scenes were successfully evaluated.")


if __name__ == "__main__":
    main()
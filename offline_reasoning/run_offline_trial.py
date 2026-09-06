"""
offline_reasoning/run_offline_trial.py

The live-camera Offline Reasoning Validation tool - autonomous version.

Run it, camera opens immediately. Press SPACE to capture a trial.

Separated scenes (1 object found):
  - Fully automatic pipeline: predict -> you confirm/correct -> saved
    image + a logged row in offline_reasoning/results/separated/<class>.csv

Cluttered scenes (2+ objects found):
  - Every detected object gets its own blue tagged box on the saved
    image. NO CSV row is written - Thami logs Ground Truth vs Predicted
    for these by hand, using the saved image + the printed list as
    reference.

Run it from the project root, venv active:
    python -m offline_reasoning.run_offline_trial
"""

from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from prompts.prompts_v1 import AGENT_ROLE, WASTE_CATEGORIES, JSON_ONLY_INSTRUCTION
from perception.gemini_classifier import classify_image, ClassificationError
from offline_reasoning.results_store import (
    append_trial, delete_trial, next_trial_number, normalize_class, read_trials,
)

WASTE_CLASSES = ["paper", "metal", "organic", "unsorted", "plastic", "glass"]

IMAGES_ROOT = Path("offline_reasoning/images")
WINDOW_NAME = "Offline Reasoning - Live Feed (SPACE=capture, Q=quit)"

# --- Annotation style, matching the reference screenshot ---
BOX_COLOR = (230, 90, 30)     # BGR - royal blue
TEXT_COLOR = (255, 255, 255)  # white
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1
TAG_PADDING = 6

# Separate from the versioned benchmark prompts in prompts_v1.py - this
# tool-specific prompt also asks for a "label" (a plain-English name),
# purely for nicer annotated images. The category we log still comes
# from "material".
ANNOTATION_PROMPT = f"""
{AGENT_ROLE} Identify all waste objects in the image.

For each object, return a JSON object with:
- "box_2d": bounding box as [ymin, xmin, ymax, xmax], normalized to 0-1000.
- "label": a short 1-3 word name identifying the object itself (e.g. "Plastic Bottle", "Drink Can").
- "material": one of {WASTE_CATEGORIES}.

{JSON_ONLY_INSTRUCTION}
"""


def denormalize_box(box_2d, width, height):
    ymin, xmin, ymax, xmax = box_2d
    x1 = int(xmin / 1000 * width)
    y1 = int(ymin / 1000 * height)
    x2 = int(xmax / 1000 * width)
    y2 = int(ymax / 1000 * height)
    return x1, y1, x2, y2


def draw_tagged_box(image, box_2d, line1, line2):
    """One blue box with a filled blue tag above it, two lines of white text."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = denormalize_box(box_2d, width, height)
    cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2, cv2.LINE_AA)

    lines = [line1, line2]
    sizes = [cv2.getTextSize(t, FONT, FONT_SCALE, FONT_THICKNESS)[0] for t in lines]
    tag_width = max(w for w, h in sizes) + TAG_PADDING * 2
    line_height = max(h for w, h in sizes) + TAG_PADDING
    tag_height = line_height * len(lines) + TAG_PADDING

    tag_x1, tag_y1 = x1, max(y1 - tag_height, 0)
    tag_x2, tag_y2 = x1 + tag_width, tag_y1 + tag_height
    cv2.rectangle(image, (tag_x1, tag_y1), (tag_x2, tag_y2), BOX_COLOR, -1)

    for i, text in enumerate(lines):
        baseline_y = tag_y1 + TAG_PADDING + (i + 1) * line_height - TAG_PADDING // 2
        cv2.putText(image, text, (tag_x1 + TAG_PADDING, baseline_y),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return image


def draw_all_detections(color_image, detections):
    """Cluttered-scene view: every detected object gets its own numbered
    blue tagged box, so the saved image lines up with the printed list."""
    annotated = color_image.copy()
    for i, det in enumerate(detections, start=1):
        label = det.get("label", det["material"].title())
        draw_tagged_box(annotated, det["box_2d"], f"{i}. {label}", f": {det['material'].title()}")
    return annotated


def confirm_ground_truth(predicted):
    """The one human touchpoint for separated trials: confirm or correct
    the true class. Returns the confirmed ground-truth class, or None if
    the trial should be aborted (bad input)."""
    raw = input(
        f"Predicted: {predicted}. Press Enter to confirm, "
        f"or type the real class if it's wrong (paper/metal/organic/unsorted/plastic/glass): "
    ).strip()
    if not raw:
        return predicted
    try:
        return normalize_class(raw)
    except ValueError:
        print(f"'{raw}' isn't one of the 6 classes - this trial was NOT recorded.")
        return None


def print_progress_tally():
    print("\nProgress so far:")
    for environment in ["separated"]:
        counts = " ".join(f"{wc}={len(read_trials(wc, environment))}" for wc in WASTE_CLASSES)
        print(f"  {environment:10s} {counts}   (auto-logged to CSV)")
    cluttered_folder = IMAGES_ROOT / "cluttered"
    cluttered_count = len(list(cluttered_folder.glob("cluttered_test*.jpg"))) if cluttered_folder.exists() else 0
    print(f"  {'cluttered':10s} {cluttered_count} scene(s) saved   (log these manually)")


def run_separated_trial(color_image, detection):
    try:
        predicted = normalize_class(detection["material"])
    except ValueError as e:
        print(f"\nGemini returned a category we don't recognize ('{detection['material']}'): {e}")
        print("This trial was NOT recorded - try the capture again.")
        return

    annotated = draw_tagged_box(
        color_image.copy(), detection["box_2d"],
        detection.get("label", predicted), f": {predicted}",
    )
    cv2.imshow(WINDOW_NAME, annotated)
    cv2.waitKey(1)

    print("\n1 object found -> environment auto-detected as 'separated'.")
    ground_truth = confirm_ground_truth(predicted)
    if ground_truth is None:
        return

    class_key = ground_truth.lower()  # fixes the capitalized-folder bug
    trial_number = next_trial_number(class_key, "separated")
    image_folder = IMAGES_ROOT / "separated" / class_key
    image_folder.mkdir(parents=True, exist_ok=True)
    image_path = image_folder / f"{class_key}_separated_test{trial_number}.jpg"
    cv2.imwrite(str(image_path), annotated)

    append_trial(class_key, "separated", ground_truth=ground_truth, predicted=predicted)

    match = "CORRECT" if ground_truth == predicted else "WRONG"
    print(f"Test Completed - Trial {trial_number}: "
          f"Ground Truth={ground_truth}, Predicted={predicted} [{match}]")
    print(f"Saved annotated image: {image_path}")

    undo = input("Press D + Enter to delete this trial, or just Enter to keep it: ").strip().lower()
    if undo == "d":
        delete_trial(class_key, "separated", trial_number)
        image_path.unlink(missing_ok=True)
        print(f"Trial {trial_number} deleted.")


def run_cluttered_capture(color_image, detections):
    annotated = draw_all_detections(color_image, detections)
    cv2.imshow(WINDOW_NAME, annotated)
    cv2.waitKey(1)

    print(f"\n{len(detections)} objects found -> environment auto-detected as 'cluttered'.")
    print("No CSV row written for cluttered scenes - log these manually:")
    for i, det in enumerate(detections, start=1):
        label = det.get("label", det["material"].title())
        print(f"  {i}. {label} -> {det['material'].title()}")

    cluttered_folder = IMAGES_ROOT / "cluttered"
    cluttered_folder.mkdir(parents=True, exist_ok=True)
    trial_number = len(list(cluttered_folder.glob("cluttered_test*.jpg"))) + 1
    image_path = cluttered_folder / f"cluttered_test{trial_number}.jpg"
    cv2.imwrite(str(image_path), annotated)
    print(f"Saved annotated image: {image_path}")

    undo = input("Press D + Enter to delete this image, or just Enter to keep it: ").strip().lower()
    if undo == "d":
        image_path.unlink(missing_ok=True)
        print(f"{image_path.name} deleted.")


def run_one_trial(color_image):
    rgb_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    try:
        detections = classify_image(rgb_image, prompt=ANNOTATION_PROMPT)
    except ClassificationError as e:
        print(f"\nCouldn't get a usable result from Gemini - this trial was NOT recorded:\n{e}")
        return

    if not detections:
        print("\nNo objects detected - make sure an item is in frame. This trial was NOT recorded.")
        return

    if len(detections) == 1:
        run_separated_trial(color_image, detections[0])
    else:
        run_cluttered_capture(color_image, detections)


def main():
    print("=== Offline Reasoning Validation - Autonomous Live Trial Tool ===\n")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    print("Live feed running. Click the camera window, then press SPACE to run a trial, or Q to quit.\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow(WINDOW_NAME, color_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == 32:  # SPACE
                run_one_trial(color_image)
                print("\nPosition the next item and press SPACE, or Q to quit.\n")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print("\nSession ended.")
    print_progress_tally()


if __name__ == "__main__":
    main()
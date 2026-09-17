"""
adaptive_validation/live_task_agent.py

No fixed task, no fixed objects. Type ANY pick-and-place instruction;
Gemini figures out object1 (pick target) and object2 (destination)
from your words, finds both live on camera, and the window shows
bounding boxes plus a trajectory line once both are found. Every box
on screen comes from a real Gemini call - no local tracker, no
classical CV standing in for it.

Gemini re-checks positions every couple of seconds (a cloud VLM call
takes real time) - between checks, the last known boxes are simply
held on screen rather than guessed at by anything else.

No 3D, no depth here on purpose. run_agent.py (Day 2/3) is separate
and untouched by this file.

Run from the project root, venv active:
    python -m adaptive_validation.live_task_agent
"""

import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from perception.gemini_classifier import classify_image, ClassificationError
from prompts.prompts_v1 import build_task_grounding_prompt, build_relocate_prompt

WINDOW_NAME = "Live Task Agent (Q to quit)"
REFRESH_INTERVAL_S = 2.0  # a Gemini call takes a couple seconds - can't do this every frame

BOX_COLOR = (230, 90, 30)         # BGR - same blue used in offline_reasoning
TRAJECTORY_COLOR = (60, 220, 60)  # BGR - bright green, visually distinct from the boxes
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1

_lock = threading.Lock()
_latest_frame = None    # most recent color image (numpy, BGR) - written by the camera thread
_latest_result = None   # {"object1": {...}, "object2": {...}} or None before a task exists
_stop_event = threading.Event()


def denormalize_box(box_2d, width, height):
    ymin, xmin, ymax, xmax = box_2d
    x1 = int(xmin / 1000 * width)
    y1 = int(ymin / 1000 * height)
    x2 = int(xmax / 1000 * width)
    y2 = int(ymax / 1000 * height)
    return x1, y1, x2, y2


def box_center(box_2d, width, height):
    x1, y1, x2, y2 = denormalize_box(box_2d, width, height)
    return (x1 + x2) // 2, (y1 + y2) // 2


def draw_tagged_box(image, box_2d, label, width, height):
    x1, y1, x2, y2 = denormalize_box(box_2d, width, height)
    cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    tag_y1 = max(y1 - th - 12, 0)
    cv2.rectangle(image, (x1, tag_y1), (x1 + tw + 12, tag_y1 + th + 10), BOX_COLOR, -1)
    cv2.putText(image, label, (x1 + 6, tag_y1 + th + 4), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


def draw_frame(color_image, result):
    annotated = color_image.copy()
    height, width = annotated.shape[:2]

    if result is None:
        cv2.putText(annotated, "Type your instruction in the terminal...", (20, 40),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        return annotated

    obj1, obj2 = result.get("object1"), result.get("object2")
    center1 = center2 = None

    if obj1 and obj1.get("found") and obj1.get("box_2d"):
        draw_tagged_box(annotated, obj1["box_2d"], f"1. {obj1['label']}", width, height)
        center1 = box_center(obj1["box_2d"], width, height)
    else:
        cv2.putText(annotated, f"'{obj1.get('label', 'object1')}' not currently visible", (20, 40),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    if obj2 and obj2.get("found") and obj2.get("box_2d"):
        draw_tagged_box(annotated, obj2["box_2d"], f"2. {obj2['label']}", width, height)
        center2 = box_center(obj2["box_2d"], width, height)
    else:
        cv2.putText(annotated, f"'{obj2.get('label', 'object2')}' not currently visible", (20, 70),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    if center1 and center2:
        cv2.arrowedLine(annotated, center1, center2, TRAJECTORY_COLOR, 3, tipLength=0.05)

    return annotated


def _camera_and_display_loop(pipeline):
    global _latest_frame
    while not _stop_event.is_set():
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        with _lock:
            _latest_frame = color_image
            result = _latest_result
        cv2.imshow(WINDOW_NAME, draw_frame(color_image, result))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            _stop_event.set()


def _detection_loop(object1_label, object2_label):
    global _latest_result
    prompt = build_relocate_prompt(object1_label, object2_label)
    while not _stop_event.is_set():
        with _lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
        if frame is not None:
            try:
                rgb_image = Image.fromarray(frame[:, :, ::-1])
                detections = classify_image(rgb_image, prompt=prompt)
                by_label = {d.get("label"): d for d in detections}
                with _lock:
                    _latest_result = {
                        "object1": by_label.get(object1_label, {"found": False, "label": object1_label}),
                        "object2": by_label.get(object2_label, {"found": False, "label": object2_label}),
                    }
            except ClassificationError as e:
                print(f"  (detection hiccup, will retry: {e})")
        time.sleep(REFRESH_INTERVAL_S)


def get_initial_task():
    while True:
        instruction = input(
            "\nWhat would you like me to do? (pick-and-place only, "
            "e.g. 'pick the banana peel and put it in the red bin'): "
        ).strip()
        if not instruction:
            continue

        with _lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
        if frame is None:
            print("  Camera isn't ready yet - one second and try again.")
            time.sleep(1)
            continue

        try:
            rgb_image = Image.fromarray(frame[:, :, ::-1])
            detections = classify_image(rgb_image, prompt=build_task_grounding_prompt(instruction))
        except ClassificationError as e:
            print(f"  Had trouble understanding the scene just now ({e}) - try again.")
            continue

        if not detections:
            print("  That doesn't read as a pick-and-place instruction - "
                  "try rephrasing (e.g. 'pick X and put it in Y').")
            continue

        by_role = {d.get("role"): d for d in detections}
        obj1, obj2 = by_role.get("object1"), by_role.get("object2")
        if obj1 is None or obj2 is None:
            print("  Got an unexpected response - try rephrasing your instruction.")
            continue

        if not obj1.get("found") or not obj2.get("found"):
            missing = obj1.get("label", "object1") if not obj1.get("found") else obj2.get("label", "object2")
            print(f"  I understood the task, but I can't currently see '{missing}' - "
                  f"make sure it's in view and try the same instruction again.")
            continue

        print(f"  Got it: pick up '{obj1['label']}' and place it at '{obj2['label']}'.")
        return obj1["label"], obj2["label"], {"object1": obj1, "object2": obj2}


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    camera_thread = threading.Thread(target=_camera_and_display_loop, args=(pipeline,), daemon=True)
    camera_thread.start()
    time.sleep(1)

    global _latest_result
    try:
        object1_label, object2_label, initial_result = get_initial_task()
        with _lock:
            _latest_result = initial_result

        detection_thread = threading.Thread(
            target=_detection_loop, args=(object1_label, object2_label), daemon=True
        )
        detection_thread.start()

        print("\nWatch the live window - it updates every couple seconds, straight from Gemini. Press Q in that window to quit.\n")
        while not _stop_event.is_set():
            time.sleep(0.2)
    finally:
        _stop_event.set()
        camera_thread.join(timeout=2)
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
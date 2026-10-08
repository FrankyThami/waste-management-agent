"""
scene_detector.py - turns one camera image into a Scene (objects + bins).

Gemini (through gemini_classifier.detect_scene) tells us WHAT is in the image
and WHERE in the image. This module converts that raw answer into the Scene
data shape from src/context/scene.py, with real pixel coordinates.

3D positions are NOT filled in here - that is Stage 2.3, which needs depth.
"""

from PIL import Image

from src.context.scene import Scene, SceneObject, SceneBin
from src.models.gemini_classifier import detect_scene
from src.utils import normalize_class


def _box_to_pixels(box_2d, width, height):
    """
    Gemini's box is [ymin, xmin, ymax, xmax] normalized 0-1000 (y first!).
    Returns (xmin, ymin, xmax, ymax) in real pixels, kept inside the image.
    Returns None if the box is not 4 numbers or has no area.
    """
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
    except (TypeError, ValueError):
        return None

    x1 = max(0, min(width - 1, round(xmin / 1000 * width)))
    x2 = max(0, min(width - 1, round(xmax / 1000 * width)))
    y1 = max(0, min(height - 1, round(ymin / 1000 * height)))
    y2 = max(0, min(height - 1, round(ymax / 1000 * height)))

    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def _box_center(box_px):
    x1, y1, x2, y2 = box_px
    return (x1 + x2) // 2, (y1 + y2) // 2


def build_scene(pil_image: Image.Image, frame_path: str | None = None) -> Scene:
    """
    Asks Gemini about one image and returns a Scene holding every valid
    object and bin it found. Anything unusable is skipped with a printed
    note, so nothing fails silently.
    """
    width, height = pil_image.size
    raw = detect_scene(pil_image)
    scene = Scene(frame_path=frame_path)

    for i, det in enumerate(raw["objects"]):
        if not isinstance(det, dict) or "material" not in det:
            print(f"[scene_detector] Skipping object {i}: no 'material' field ({det})")
            continue
        try:
            waste_class = normalize_class(det["material"]).lower()
        except ValueError:
            print(f"[scene_detector] Skipping object {i}: unrecognized class '{det['material']}'")
            continue
        box_px = _box_to_pixels(det.get("box_2d"), width, height)
        if box_px is None:
            print(f"[scene_detector] Skipping object {i}: unusable box ({det.get('box_2d')})")
            continue
        scene.objects.append(SceneObject(
            object_id=f"obj_{i}",
            waste_class=waste_class,
            box_px=box_px,
            center_px=_box_center(box_px),
        ))

    for i, det in enumerate(raw["bins"]):
        if not isinstance(det, dict):
            print(f"[scene_detector] Skipping bin {i}: not a valid entry ({det})")
            continue
        colour = str(det.get("colour") or det.get("color") or "").strip().lower()
        if not colour:
            print(f"[scene_detector] Skipping bin {i}: no colour given ({det})")
            continue
        box_px = _box_to_pixels(det.get("box_2d"), width, height)
        if box_px is None:
            print(f"[scene_detector] Skipping bin {i}: unusable box ({det.get('box_2d')})")
            continue
        scene.bins.append(SceneBin(
            bin_id=f"bin_{i}",
            colour=colour,
            box_px=box_px,
            center_px=_box_center(box_px),
        ))

    return scene
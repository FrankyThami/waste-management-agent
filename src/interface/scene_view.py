"""
scene_view.py - draws a Scene on top of the camera image so a human can check it.

Objects: blue box + blue tag with white text (same style as the offline trials).
Bins:    orange box + orange tag with white text, so they are easy to tell apart.
A small cross marks each centre point.
"""

import cv2

from src.context.scene import Scene

BLUE = (255, 0, 0)       # OpenCV uses BGR order, so this is blue
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)


def _draw_box_with_tag(image, box_px, center_px, label, colour):
    x1, y1, x2, y2 = box_px
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    cv2.drawMarker(image, center_px, colour, cv2.MARKER_CROSS, 16, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

    # The tag sits just above the box; if the box touches the top edge, put it inside.
    tag_top = y1 - text_h - baseline - 6
    if tag_top < 0:
        tag_top = y1
    tag_bottom = tag_top + text_h + baseline + 6

    cv2.rectangle(image, (x1, tag_top), (x1 + text_w + 8, tag_bottom), colour, -1)
    cv2.putText(image, label, (x1 + 4, tag_bottom - baseline - 3),
                font, scale, WHITE, thickness, cv2.LINE_AA)


def draw_scene(color_bgr, scene: Scene):
    """Returns a copy of the camera image with every object and bin drawn on it."""
    annotated = color_bgr.copy()
    for obj in scene.objects:
        _draw_box_with_tag(annotated, obj.box_px, obj.center_px,
                           f"{obj.object_id} {obj.waste_class}", BLUE)
    for b in scene.bins:
        _draw_box_with_tag(annotated, b.box_px, b.center_px,
                           f"{b.bin_id} {b.colour} bin", ORANGE)
    return annotated
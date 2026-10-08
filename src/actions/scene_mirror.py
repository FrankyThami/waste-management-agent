"""
scene_mirror.py - shows the real table in the MATLAB window.

After every look at the table, the bins the camera found are drawn in MATLAB as
coloured boxes at their measured positions: a "digital twin" (a virtual copy) of
the table. It is DISPLAY ONLY - nothing here changes what the agent decides or how
the arm moves. It also does nothing unless HAL_MODE=matlab.
"""

from src.actions.hal import HAL_MODE, show_scene_boxes

# Real size of one bin in metres, measured by hand. This is more accurate than the
# camera's loose boxes, so the camera is only trusted for WHERE the bin is.
BIN_LONG_M = 0.15
BIN_SHORT_M = 0.11
BIN_HEIGHT_M = 0.056

# Colour names the agent can report -> (red, green, blue), each 0..1, used for drawing.
COLOUR_RGB = {
    "red": (0.85, 0.15, 0.15),
    "green": (0.15, 0.65, 0.25),
    "blue": (0.15, 0.35, 0.85),
    "yellow": (0.95, 0.80, 0.10),
    "orange": (0.95, 0.55, 0.10),
    "purple": (0.55, 0.25, 0.75),
    "pink": (0.95, 0.50, 0.70),
    "brown": (0.55, 0.35, 0.20),
    "black": (0.10, 0.10, 0.10),
    "white": (0.95, 0.95, 0.95),
    "grey": (0.55, 0.55, 0.55),
    "gray": (0.55, 0.55, 0.55),
}
UNKNOWN_RGB = (0.60, 0.60, 0.60)   # any colour name not in the table above is drawn grey


def _make_box(bin_info):
    """
    One bin from observe_scene -> one box for MATLAB.

    Which way does the bin lie? The camera only gives an upright pixel box, with no angle.
    In the robot frame the image-vertical direction is x and the image-horizontal direction
    is y (see frame_transform.py), so: if the pixel box is taller than it is wide, the long
    (15 cm) side lies along x (yaw 0); otherwise along y (yaw 90). Rough on purpose.
    """
    position = bin_info["position"]
    yaw = 0.0 if bin_info["height_m"] >= bin_info["width_m"] else 90.0
    return {
        "name": bin_info["colour"],
        "x": position["x"],
        "y": position["y"],
        "yawDeg": yaw,
        "lengthM": BIN_LONG_M,
        "widthM": BIN_SHORT_M,
        "heightM": BIN_HEIGHT_M,
        "rgb": list(COLOUR_RGB.get(bin_info["colour"], UNKNOWN_RGB)),
    }


def mirror_bins(bins):
    """
    Draws the bins from one observation (the "bins" list that observe() builds).
    Never raises: a drawing problem must not stop the agent.
    """
    if HAL_MODE != "matlab":
        return
    try:
        boxes = [_make_box(b) for b in bins]
        if not show_scene_boxes(boxes):
            print("[scene_mirror] MATLAB did not draw the boxes (the agent carries on)")
            return
        for b in bins:
            p = b["position"]
            # The measured rim height should come out close to the real 5.6 cm. A big
            # difference means the table-plane calibration is off.
            print(f"[scene_mirror] {b['colour']} bin drawn at x={p['x']:.3f} y={p['y']:.3f} m "
                  f"(measured rim height {p['z']:.3f} m, real {BIN_HEIGHT_M:.3f} m)")
    except Exception as error:   # display only - never let it break a run
        print(f"[scene_mirror] Could not draw the boxes (the agent carries on): {error}")
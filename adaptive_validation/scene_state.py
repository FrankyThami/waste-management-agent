"""
adaptive_validation/scene_state.py

Runs the RealSense camera on its own background thread that never
stops pulling frames, and shows a live preview window for the whole
run - including during a pause. This does two things at once: you can
SEE exactly what the camera sees before pressing Enter, and because
frames are continuously drained in the background, capture_and_store()
always hands back the current frame - never something stuck in a
backlog from before the pause.
"""

import threading

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

_pipeline = None
_align = None
_thread = None
_stop_event = None
_lock = threading.Lock()

_latest_color_frame = None
_latest_depth_frame = None

WINDOW_NAME = "Live camera - Waste Management Agent"

last_color_frame = None
last_depth_frame = None
last_rgb_image = None
last_detections = None  # set by observe_scene/reobserve_scene, in tools.py
last_known_positions = {}  # target_description -> [x, y, z] robot frame, meters - set by locate_object()


def _camera_loop():
    global _latest_color_frame, _latest_depth_frame
    while not _stop_event.is_set():
        frames = _pipeline.wait_for_frames()
        aligned = _align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # these frames are about to be read from a DIFFERENT thread
        # than the one that received them - .keep() stops RealSense
        # from recycling their memory once this loop iteration ends
        color_frame.keep()
        depth_frame.keep()

        with _lock:
            _latest_color_frame = color_frame
            _latest_depth_frame = depth_frame

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow(WINDOW_NAME, color_image)
        cv2.waitKey(1)  # lets the OS actually paint the window


def start_camera():
    global _pipeline, _align, _thread, _stop_event
    if _pipeline is not None:
        return
    _pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    _pipeline.start(config)
    _align = rs.align(rs.stream.color)

    _stop_event = threading.Event()
    _thread = threading.Thread(target=_camera_loop, daemon=True)
    _thread.start()


def stop_camera():
    global _pipeline
    if _stop_event is not None:
        _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=2)
    cv2.destroyAllWindows()
    if _pipeline is not None:
        _pipeline.stop()
        _pipeline = None


def capture_and_store():
    """Return whatever the background thread most recently captured -
    always current, never stale, since the camera never stops
    pulling frames in the background."""
    global last_color_frame, last_depth_frame, last_rgb_image
    with _lock:
        if _latest_color_frame is None:
            raise RuntimeError("No frames yet - camera still warming up, try again in a moment.")
        last_color_frame = _latest_color_frame
        last_depth_frame = _latest_depth_frame

    color_image = np.asanyarray(last_color_frame.get_data())
    last_rgb_image = Image.fromarray(color_image[:, :, ::-1])
    return last_rgb_image
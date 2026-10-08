"""
camera.py - grabs one aligned colour + depth frame from the RealSense D435i.

Depth is measured over several frames and combined (per-pixel median of the
valid readings), so a momentary dropout on a glossy or transparent item does
not turn into "no depth". A pixel with too few valid readings stays 0.0 -
we never invent depth.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

WARMUP_FRAMES = 15      # let auto-exposure settle before measuring
DEPTH_FRAMES = 10       # depth is combined over this many frames
MIN_VALID_FRAMES = 3    # a pixel needs a real reading in at least this many frames


@dataclass
class CapturedFrame:
    color_bgr: np.ndarray    # colour image in OpenCV (BGR) format, shape (height, width, 3)
    depth_m: np.ndarray      # depth in METRES for every colour pixel, shape (height, width); 0.0 = no reading
    intrinsics: object       # camera lens model, needed to turn pixel + depth into 3D

    @property
    def width(self) -> int:
        return self.color_bgr.shape[1]

    @property
    def height(self) -> int:
        return self.color_bgr.shape[0]

    def to_pil(self) -> Image.Image:
        """The colour image as a PIL image (the format we send to Gemini)."""
        return Image.fromarray(cv2.cvtColor(self.color_bgr, cv2.COLOR_BGR2RGB))


def _combine_depth(depth_stack):
    """Per-pixel median of the valid (non-zero) readings across all frames; 0.0 where too few are valid."""
    stack = np.stack(depth_stack)                       # shape (frames, height, width), metres
    stack[stack == 0] = np.nan                          # 0 means "no reading" - ignore it
    stack.sort(axis=0)                                  # NaN values sort to the end
    valid_count = np.sum(~np.isnan(stack), axis=0)      # valid readings per pixel
    middle = np.maximum(valid_count - 1, 0) // 2        # index of the median among the valid ones
    combined = np.take_along_axis(stack, middle[None, :, :], axis=0)[0]
    combined[valid_count < MIN_VALID_FRAMES] = 0.0
    return np.nan_to_num(combined, nan=0.0)


def capture_frame() -> CapturedFrame:
    """Starts the D435i, lets auto-exposure settle, grabs several frames, stops the camera."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()  # metres per depth unit
    align = rs.align(rs.stream.color)

    depth_stack = []
    color_bgr = None
    intrinsics = None
    try:
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()

        for _ in range(DEPTH_FRAMES):
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            depth_stack.append(np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale)
            color_bgr = np.asanyarray(color_frame.get_data()).copy()
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    finally:
        pipeline.stop()

    if len(depth_stack) < MIN_VALID_FRAMES:
        raise RuntimeError("RealSense did not return enough valid frames.")

    return CapturedFrame(color_bgr=color_bgr, depth_m=_combine_depth(depth_stack), intrinsics=intrinsics)
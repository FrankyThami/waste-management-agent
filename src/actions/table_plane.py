"""
table_plane.py - finds the tabletop as a flat plane in the camera's 3D view (Stage 2.3b).

Why: the camera never looks perfectly straight down, so "distance from the camera"
is not the same as "height above the table". Fitting a plane to the bare table gives:
  - the camera's tilt,
  - the camera's height above the table,
  - a way to turn any 3D point into a true height above the table.

Camera frame (RealSense): x = right, y = down, z = distance from the camera. Metres.
"""

from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs

SAMPLE_STRIDE = 8                           # use every 8th pixel in each direction (fast, ~14,000 points)
BOX_MARGIN_PX = 12                          # stay this far away from detected boxes so item edges don't leak in
TRIM_THRESHOLDS_M = (0.02, 0.008, 0.005)    # progressively stricter outlier trimming (metres)
MIN_PLANE_POINTS = 2000                     # need at least this many bare-table points for a trustworthy fit


@dataclass
class TablePlane:
    """The tabletop as a plane: normal . p = d for every point p on the table."""
    normal: tuple[float, float, float]   # unit vector pointing from the table UP toward the camera
    d: float                             # plane offset, metres
    n_points: int                        # how many table points were used
    rms_m: float                         # how far the table points scatter around the plane (metres)

    def height_above(self, point) -> float:
        """True height (metres) of a camera-frame point above the table plane."""
        return float(np.dot(self.normal, point) - self.d)

    @property
    def camera_height_m(self) -> float:
        """Distance from the camera to the table plane."""
        return abs(self.d)

    @property
    def tilt_deg(self) -> float:
        """Angle between the camera's viewing direction and the table's normal (0 = looking straight down)."""
        return float(np.degrees(np.arccos(np.clip(-self.normal[2], -1.0, 1.0))))


def _table_points(depth_m, intrinsics, exclude_boxes):
    """3D points (camera frame) from a sparse grid of pixels that have depth and lie outside every box."""
    height, width = depth_m.shape
    keep = np.zeros((height, width), dtype=bool)
    keep[::SAMPLE_STRIDE, ::SAMPLE_STRIDE] = True
    keep &= depth_m > 0
    for x1, y1, x2, y2 in exclude_boxes:
        keep[max(0, y1 - BOX_MARGIN_PX):y2 + BOX_MARGIN_PX + 1,
             max(0, x1 - BOX_MARGIN_PX):x2 + BOX_MARGIN_PX + 1] = False
    rows, cols = np.nonzero(keep)
    points = [
        rs.rs2_deproject_pixel_to_point(intrinsics, [float(c), float(r)], float(depth_m[r, c]))
        for r, c in zip(rows, cols)
    ]
    return np.array(points)


def _fit_plane(points):
    """Best-fit plane through points: returns (unit normal, centroid)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    return vt[-1], centroid   # the direction with the LEAST spread is the plane's normal


def fit_table_plane(depth_m, intrinsics, exclude_boxes=()):
    """
    Fits the tabletop plane from every depth pixel outside `exclude_boxes`
    (a list of (x1, y1, x2, y2) pixel boxes of objects and bins).
    Returns a TablePlane, or None if too little bare table is visible.
    """
    points = _table_points(depth_m, intrinsics, exclude_boxes)
    if len(points) < MIN_PLANE_POINTS:
        return None

    # Fit, throw away points far from the plane, fit again - stricter each round.
    kept = points
    for threshold in TRIM_THRESHOLDS_M:
        normal, centroid = _fit_plane(kept)
        distances = np.abs((points - centroid) @ normal)
        kept = points[distances < threshold]
        if len(kept) < MIN_PLANE_POINTS:
            return None

    normal, centroid = _fit_plane(kept)
    if normal[2] > 0:       # make the normal point toward the camera (z is distance away from it)
        normal = -normal
    d = float(normal @ centroid)
    rms = float(np.sqrt(np.mean(((kept - centroid) @ normal) ** 2)))
    return TablePlane(normal=tuple(float(v) for v in normal), d=d, n_points=len(kept), rms_m=rms)
"""Geometric signals computed from MediaPipe face landmarks.

All functions take an (N, 2) or (N, 3) array of landmark coordinates in pixel
space (z scaled like x) and return plain floats, so they are easy to unit test
without a camera.
"""

from __future__ import annotations

import math

import numpy as np

# MediaPipe Face Mesh indices. Order for each eye: outer corner, upper lid x2,
# inner corner, lower lid x2 (p1..p6 in the Soukupova & Cech EAR paper).
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

# Eye corners ordered image-left -> image-right, and iris centres (refined mesh).
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)
RIGHT_IRIS, LEFT_IRIS = 468, 473

# Mouth: two corners, then three upper/lower pairs across the inner lips.
MOUTH_CORNERS = (61, 291)
MOUTH_PAIRS = ((81, 178), (13, 14), (311, 402))

# Head axes: outer eye corners (left-right) and forehead -> chin (top-bottom).
FOREHEAD, CHIN = 10, 152

# Full contours, only used for drawing.
RIGHT_EYE_CONTOUR = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
LEFT_EYE_CONTOUR = (362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382)
MOUTH_CONTOUR = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def eye_aspect_ratio(pts: np.ndarray, idx: tuple[int, ...]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|). Drops towards 0 as the eye closes."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    width = _dist(p1, p4)
    if width < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * width)


def mean_ear(pts: np.ndarray) -> float:
    return (eye_aspect_ratio(pts, RIGHT_EYE) + eye_aspect_ratio(pts, LEFT_EYE)) / 2.0


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """Mean inner-lip opening divided by mouth width. Rises sharply during a yawn."""
    width = _dist(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]])
    if width < 1e-6:
        return 0.0
    opening = sum(_dist(pts[a], pts[b]) for a, b in MOUTH_PAIRS) / len(MOUTH_PAIRS)
    return opening / width


def head_pose(pts: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from 3D landmarks.

    Builds the face's own axes (across the eyes, forehead to chin) and takes the
    forward normal. This has no ±180° wrap-around, unlike decomposing a full
    transform matrix. Conventions, in image space:
      yaw   > 0  face turned towards the right side of the image
      pitch < 0  looking down
      roll  > 0  head tilted clockwise in the image
    """
    p = np.asarray(pts, dtype=float)
    right = p[LEFT_EYE_CORNERS[1]] - p[RIGHT_EYE_CORNERS[0]]
    down = p[CHIN] - p[FOREHEAD]
    fwd = np.cross(down, right)  # points out of the face, towards the camera
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return 0.0, 0.0, 0.0
    fx, fy, fz = fwd / n
    # Camera looks along +z (MediaPipe z grows away from the camera), so a
    # frontal face has fwd = (0, 0, -1).
    yaw = math.degrees(math.atan2(fx, -fz))
    pitch = math.degrees(math.atan2(-fy, math.hypot(fx, fz)))
    roll = math.degrees(math.atan2(right[1], right[0]))
    return pitch, yaw, roll


def gaze_ratio(pts: np.ndarray) -> float | None:
    """Horizontal iris position within the eyes, -1 (image left) .. +1 (image right).

    Requires the 478-point mesh with iris landmarks. 0 means centred.
    """
    if len(pts) < 478:
        return None
    vals = []
    for (a, b), iris in ((RIGHT_EYE_CORNERS, RIGHT_IRIS), (LEFT_EYE_CORNERS, LEFT_IRIS)):
        pa, pb, pi = pts[a][:2], pts[b][:2], pts[iris][:2]
        axis = pb - pa
        denom = float(axis @ axis)
        if denom < 1e-6:
            continue
        t = float((pi - pa) @ axis) / denom
        vals.append((t - 0.5) * 2.0)
    return float(np.clip(np.mean(vals), -1, 1)) if vals else None


def face_box(pts: np.ndarray) -> tuple[float, float, float, float]:
    xs, ys = pts[:, 0], pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

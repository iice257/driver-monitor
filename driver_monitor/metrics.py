"""Geometric signals computed from MediaPipe face landmarks.

All functions take an (N, 2) or (N, 3) array of landmark coordinates in pixel
space and return plain floats, so they are easy to unit test without a camera.
"""

from __future__ import annotations

import math

import numpy as np

# MediaPipe Face Mesh indices. Order for each eye: outer corner, upper lid x2,
# inner corner, lower lid x2 (p1..p6 in the Soukupova & Cech EAR paper).
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

# Mouth: two corners, then three upper/lower pairs across the inner lips.
MOUTH_CORNERS = (61, 291)
MOUTH_PAIRS = ((81, 178), (13, 14), (311, 402))

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


def euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from a 3x3 rotation or 4x4 transform matrix.

    Positive yaw = head turned to the camera's right, positive pitch = chin up.
    """
    r = np.asarray(m, dtype=float)[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock, will not happen for a driver facing a camera
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

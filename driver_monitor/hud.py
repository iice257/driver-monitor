"""On-frame overlay: landmark contours, a translucent telemetry panel and alert banner."""

from __future__ import annotations

import cv2
import numpy as np

from . import metrics
from .detector import FaceReading
from .state import Level, Status

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

# BGR
WHITE = (245, 245, 245)
MUTED = (165, 165, 165)
TRACK = (70, 70, 70)
LEVEL_COLOR = {
    Level.OK: (120, 200, 90),
    Level.WARN: (40, 180, 250),
    Level.ALERT: (60, 60, 235),
}
LEVEL_LABEL = {Level.OK: "ATTENTIVE", Level.WARN: "CAUTION", Level.ALERT: "ALERT"}


def _blend_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color, alpha: float) -> None:
    roi = img[y0:y1, x0:x1]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _contour(img: np.ndarray, pts: np.ndarray, idx: tuple[int, ...], color) -> None:
    poly = pts[list(idx), :2].astype(np.int32)
    cv2.polylines(img, [poly], True, color, 1, AA)


def _bar(img, x, y, w, label, value, lo, hi, marker=None, color=WHITE) -> int:
    cv2.putText(img, label, (x, y), FONT, 0.42, MUTED, 1, AA)
    cv2.putText(img, value_text(value), (x + w - 48, y), FONT, 0.42, WHITE, 1, AA)
    by = y + 9
    cv2.line(img, (x, by), (x + w, by), TRACK, 4, AA)
    frac = float(np.clip((value - lo) / (hi - lo), 0, 1))
    if lo < 0 < hi:  # signed value: fill outwards from the centre
        cx = x + int(w * -lo / (hi - lo))
        cv2.line(img, (cx, by - 5), (cx, by + 5), MUTED, 1, AA)
        cv2.line(img, (cx, by), (x + int(w * frac), by), color, 4, AA)
    else:
        cv2.line(img, (x, by), (x + int(w * frac), by), color, 4, AA)
    if marker is not None:
        mx = x + int(w * float(np.clip((marker - lo) / (hi - lo), 0, 1)))
        cv2.line(img, (mx, by - 6), (mx, by + 6), WHITE, 1, AA)
    return y + 34


def value_text(v: float) -> str:
    return f"{v:5.2f}" if abs(v) < 10 else f"{v:5.0f}"


def draw(img: np.ndarray, face: FaceReading | None, s: Status, fps: float) -> np.ndarray:
    h, w = img.shape[:2]
    color = LEVEL_COLOR[s.level]

    if face is not None:
        eye_col = (60, 60, 235) if s.eyes_closed else (200, 230, 120)
        _contour(img, face.points, metrics.RIGHT_EYE_CONTOUR, eye_col)
        _contour(img, face.points, metrics.LEFT_EYE_CONTOUR, eye_col)
        _contour(img, face.points, metrics.MOUTH_CONTOUR, (200, 200, 200))

    # Telemetry panel
    pw, pad = 230, 18
    x0, y0 = w - pw - 16, 16
    x1, y1 = w - 16, min(h - 16, y0 + 262)
    _blend_rect(img, x0, y0, x1, y1, (18, 18, 18), 0.62)

    x, y = x0 + pad, y0 + 30
    label = "CALIBRATING" if s.calibrating else LEVEL_LABEL[s.level]
    cv2.circle(img, (x + 5, y - 5), 5, color if not s.calibrating else MUTED, -1, AA)
    cv2.putText(img, label, (x + 18, y), FONT, 0.55, WHITE, 1, AA)
    cv2.putText(img, f"{fps:4.0f} fps", (x1 - pad - 58, y), FONT, 0.4, MUTED, 1, AA)
    y += 30

    bw = pw - 2 * pad
    ear = face.ear if face else 0.0
    mar = face.mar if face else 0.0
    yaw = face.yaw if face else 0.0
    pitch = face.pitch if face else 0.0
    y = _bar(img, x, y, bw, "Eye openness (EAR)", ear, 0.0, 0.45, s.ear_threshold,
             (60, 60, 235) if s.eyes_closed else (200, 230, 120))
    y = _bar(img, x, y, bw, "Mouth opening (MAR)", mar, 0.0, 1.0)
    y = _bar(img, x, y, bw, "PERCLOS (60s)", s.perclos, 0.0, 0.4, 0.15)
    y = _bar(img, x, y, bw, "Head yaw", yaw, -60, 60)
    y = _bar(img, x, y, bw, "Head pitch", pitch, -45, 45)
    cv2.putText(img, f"Yawns (5 min)   {s.yawns}", (x, y), FONT, 0.42, MUTED, 1, AA)

    # Alert banner and frame border
    if s.level > Level.OK and s.reasons:
        text = "  /  ".join(s.reasons)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
        bx0, by1 = 16, h - 16
        _blend_rect(img, bx0, by1 - th - 26, min(w - 16, bx0 + tw + 32), by1, color, 0.85)
        cv2.putText(img, text, (bx0 + 16, by1 - 13), FONT, 0.6, WHITE, 1, AA)
    if s.level == Level.ALERT:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 6)

    if s.calibrating:
        msg = "Calibrating: look at the road as you normally would"
        (tw, th), _ = cv2.getTextSize(msg, FONT, 0.55, 1)
        by1 = h - 16
        _blend_rect(img, 16, by1 - th - 26, min(w - 16, 16 + tw + 32), by1, (18, 18, 18), 0.62)
        cv2.putText(img, msg, (32, by1 - 13), FONT, 0.55, WHITE, 1, AA)
    return img

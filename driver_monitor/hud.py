"""On-frame overlay: landmark contours, detections, a telemetry panel and alert banner."""

from __future__ import annotations

import cv2
import numpy as np

from . import metrics
from .detector import FaceReading
from .phone import PhoneResult
from .state import Level, Status

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

# BGR
WHITE = (245, 245, 245)
MUTED = (165, 165, 165)
TRACK = (70, 70, 70)
GOOD = (200, 230, 120)
LEVEL_COLOR = {
    Level.OK: (120, 200, 90),
    Level.WARN: (40, 180, 250),
    Level.ALERT: (60, 60, 235),
}
LEVEL_LABEL = {Level.OK: "ATTENTIVE", Level.WARN: "CAUTION", Level.ALERT: "ALERT"}
HAND_LINKS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))


def _blend_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color, alpha: float) -> None:
    x0, y0, x1, y1 = max(0, x0), max(0, y0), max(0, x1), max(0, y1)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _contour(img: np.ndarray, pts: np.ndarray, idx: tuple[int, ...], color) -> None:
    poly = pts[list(idx), :2].astype(np.int32)
    cv2.polylines(img, [poly], True, color, 1, AA)


def _fmt(v: float) -> str:
    return f"{v:5.2f}" if abs(v) < 10 else f"{v:5.0f}"


def _bar(img, x, y, w, label, value, lo, hi, marker=None, color=WHITE, active=False) -> int:
    cv2.putText(img, label, (x, y), FONT, 0.42, WHITE if active else MUTED, 1, AA)
    cv2.putText(img, _fmt(value), (x + w - 48, y), FONT, 0.42, WHITE, 1, AA)
    by = y + 9
    cv2.line(img, (x, by), (x + w, by), TRACK, 4, AA)
    frac = float(np.clip((value - lo) / (hi - lo), 0, 1))
    if lo < 0 < hi:  # signed value: fill outwards from the centre
        cx = x + int(w * -lo / (hi - lo))
        cv2.line(img, (cx, by - 5), (cx, by + 5), MUTED, 1, AA)
        cv2.line(img, (cx, by), (x + int(w * frac), by), color, 4, AA)
    else:
        cv2.line(img, (x, by), (x + int(w * frac), by), color, 4, AA)
    for m in (marker if isinstance(marker, (list, tuple)) else [marker] if marker is not None else []):
        mx = x + int(w * float(np.clip((m - lo) / (hi - lo), 0, 1)))
        cv2.line(img, (mx, by - 6), (mx, by + 6), WHITE, 1, AA)
    return y + 34


def _detections(img: np.ndarray, phone: PhoneResult) -> None:
    for (x0, y0, x1, y1), score in phone.phones:
        c = LEVEL_COLOR[Level.ALERT]
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, 2, AA)
        cv2.putText(img, f"phone {score:.0%}", (int(x0), int(y0) - 6), FONT, 0.45, c, 1, AA)
    for hand in phone.hands:
        p = hand.astype(np.int32)
        for a, b in HAND_LINKS:
            cv2.line(img, tuple(p[a]), tuple(p[b]), (210, 170, 90), 1, AA)


def draw(img: np.ndarray, face: FaceReading | None, s: Status, fps: float,
         phone: PhoneResult | None = None, cfg=None) -> np.ndarray:
    h, w = img.shape[:2]
    color = LEVEL_COLOR[s.level]

    if phone is not None:
        _detections(img, phone)
    if face is not None:
        eye_col = (60, 60, 235) if s.eyes_closed else GOOD
        _contour(img, face.points, metrics.RIGHT_EYE_CONTOUR, eye_col)
        _contour(img, face.points, metrics.LEFT_EYE_CONTOUR, eye_col)
        _contour(img, face.points, metrics.MOUTH_CONTOUR, (60, 180, 250) if s.yawning else (200, 200, 200))
        for i in (metrics.RIGHT_IRIS, metrics.LEFT_IRIS):
            cv2.circle(img, tuple(face.points[i, :2].astype(int)), 2, WHITE, -1, AA)

    # Telemetry panel
    pw, pad, rows = 236, 18, 7
    x0, y0 = w - pw - 16, 16
    x1, y1 = w - 16, min(h - 16, y0 + 62 + rows * 34 + 10)
    _blend_rect(img, x0, y0, x1, y1, (18, 18, 18), 0.62)

    x, y = x0 + pad, y0 + 30
    label = "CALIBRATING" if s.calibrating else LEVEL_LABEL[s.level]
    cv2.circle(img, (x + 5, y - 5), 5, MUTED if s.calibrating else color, -1, AA)
    cv2.putText(img, label, (x + 18, y), FONT, 0.55, WHITE, 1, AA)
    cv2.putText(img, f"{fps:4.0f} fps", (x1 - pad - 58, y), FONT, 0.4, MUTED, 1, AA)
    y += 30

    bw = pw - 2 * pad
    yl = cfg.yaw_limit_deg if cfg else 25
    pd = cfg.pitch_down_limit_deg if cfg else 15
    gl = cfg.gaze_limit if cfg else 0.45
    jaw = face.jaw_open if face else 0.0
    y = _bar(img, x, y, bw, "Eye openness (EAR)", face.ear if face else 0.0, 0.0, 0.45, s.ear_threshold,
             (60, 60, 235) if s.eyes_closed else GOOD, s.eyes_closed)
    y = _bar(img, x, y, bw, "Jaw open", jaw, 0.0, 1.0, cfg.yawn_jaw if cfg else 0.45,
             (60, 180, 250) if s.yawning else WHITE, s.yawning)
    y = _bar(img, x, y, bw, "PERCLOS (60s)", s.perclos, 0.0, 0.4, cfg.perclos_warn if cfg else 0.15)
    y = _bar(img, x, y, bw, "Head yaw", s.yaw, -60, 60, (-yl, yl), active=abs(s.yaw) > yl)
    y = _bar(img, x, y, bw, "Head pitch", s.pitch, -45, 45, -pd, active=s.pitch < -pd)
    y = _bar(img, x, y, bw, "Gaze", s.gaze, -1, 1, (-gl, gl), active=abs(s.gaze) > gl)
    n_phones = len(phone.phones) if phone else 0
    cv2.putText(img, f"Yawns (5 min)  {s.yawns}", (x, y), FONT, 0.42, MUTED, 1, AA)
    cv2.putText(img, f"Phone  {'yes' if n_phones else 'no'}", (x + bw - 70, y), FONT, 0.42,
                LEVEL_COLOR[Level.ALERT] if n_phones else MUTED, 1, AA)

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

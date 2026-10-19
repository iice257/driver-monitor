"""On-frame overlay: landmark contours, detections, a telemetry panel and alert banner.

Shapes are drawn with OpenCV; all text is set in Inter (bundled in driver_monitor/fonts, SIL OFL)
with Pillow/FreeType, so it is crisp at any resolution. Everything scales with the frame height
(designed at 720p), and the panel never leaves the frame.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import metrics
from .alerts import cue_for
from .detector import FaceReading
from .phone import PhoneResult
from .state import Level, Status

AA = cv2.LINE_AA
FONT_FILE = Path(__file__).resolve().parent / "fonts" / "Inter.ttf"

# BGR, matching the web inspector's palette
FG = (240, 242, 242)
FG2 = (161, 166, 166)
FG3 = (104, 108, 108)
TRACK = (66, 64, 64)
PANEL = (18, 17, 17)
GOOD = (138, 208, 95)
WARN = (70, 181, 245)
ALERT = (68, 68, 239)
LEVEL_COLOR = {Level.OK: GOOD, Level.WARN: WARN, Level.ALERT: ALERT}
LEVEL_LABEL = {Level.OK: "Attentive", Level.WARN: "Caution", Level.ALERT: "Alert"}
HAND_LINKS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))

REGULAR, MEDIUM, SEMIBOLD, BOLD = 400, 500, 600, 700


@lru_cache(maxsize=64)
def _font(px: int, weight: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT_FILE), px)
    f.set_variation_by_axes([min(32, max(14, px)), weight])
    return f


def text_width(s: str, px: int, weight: int = REGULAR) -> float:
    return _font(px, weight).getlength(s)


def draw_text(img: np.ndarray, items: list[tuple]) -> None:
    """Render text items onto img in place. Item: (x, baseline_y, text, px, weight, bgr, align) with align l/r/m."""
    if not items:
        return
    placed = []
    for x, y, s, px, wt, col, align in items:
        w = _font(px, wt).getlength(s)
        left = x - (w if align == "r" else w / 2 if align == "m" else 0)
        placed.append((left, y, s, px, wt, col, w))
    x0 = int(min(p[0] for p in placed)) - 4
    y0 = int(min(p[1] - p[3] * 1.1 for p in placed)) - 2
    x1 = int(max(p[0] + p[6] for p in placed)) + 4
    y1 = int(max(p[1] + p[3] * 0.35 for p in placed)) + 2
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = Image.fromarray(np.ascontiguousarray(img[y0:y1, x0:x1, ::-1]))
    d = ImageDraw.Draw(roi)
    for left, y, s, px, wt, col, _ in placed:
        d.text((left - x0, y - y0), s, font=_font(px, wt), fill=col[::-1], anchor="ls")
    img[y0:y1, x0:x1] = np.asarray(roi)[:, :, ::-1]


def _rounded_blend(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int, color, alpha: float) -> None:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    rw, rh = x1 - x0, y1 - y0
    r = min(r, rw // 2, rh // 2)
    cv2.rectangle(mask, (r, 0), (rw - r - 1, rh - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (rw - 1, rh - r - 1), 255, -1)
    for cx, cy in ((r, r), (rw - r - 1, r), (r, rh - r - 1), (rw - r - 1, rh - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1, AA)
    m = (mask.astype(np.float32) / 255.0 * alpha)[:, :, None]
    roi[:] = (roi * (1 - m) + np.array(color, np.float32) * m).astype(np.uint8)


def _blend_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color, alpha: float) -> None:
    _rounded_blend(img, x0, y0, x1, y1, 0, color, alpha)


def _contour(img: np.ndarray, pts: np.ndarray, idx: tuple[int, ...], color) -> None:
    poly = pts[list(idx), :2].astype(np.int32)
    cv2.polylines(img, [poly], True, color, 1, AA)


def _fmt(key: str, v: float) -> str:
    if key in ("yaw", "pitch"):
        return f"{round(v)}°" if abs(v) >= 0.5 else "0°"  # never "-0°"
    if key == "perclos":
        return f"{round(v * 100)}%"
    return f"{v:.2f}" if abs(v) >= 0.005 else "0.00"


def _bar(img, x: int, y: int, w: int, k: float, value: float, lo: float, hi: float, marks, color) -> None:
    """A rounded track with the value filled from the left (or outwards from the centre when signed) and threshold ticks."""
    th = max(3, round(4 * k))
    frac = float(np.clip((value - lo) / (hi - lo), 0, 1))
    cv2.line(img, (x, y), (x + w, y), TRACK, th, AA)
    for ex in (x, x + w):
        cv2.circle(img, (ex, y), th // 2, TRACK, -1, AA)
    if lo < 0 < hi:
        cx = x + int(w * -lo / (hi - lo))
        a, b = sorted((cx, x + int(w * frac)))
        if b > a:
            cv2.line(img, (a, y), (b, y), color, th, AA)
        cv2.line(img, (cx, y - th), (cx, y + th), FG3, 1, AA)
    elif frac > 0:
        cv2.line(img, (x, y), (x + int(w * frac), y), color, th, AA)
        cv2.circle(img, (x, y), th // 2, color, -1, AA)
    for m in marks:
        mx = x + int(w * float(np.clip((m - lo) / (hi - lo), 0, 1)))
        cv2.line(img, (mx, y - th - 1), (mx, y + th + 1), FG, max(1, round(k)), AA)


def _detections(img: np.ndarray, phone: PhoneResult, k: float) -> None:
    labels = []
    for (x0, y0, x1, y1), score in phone.phones:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), ALERT, 2, AA)
        labels.append((int(x0), max(int(14 * k), int(y0) - 8), f"Phone {score:.0%}", round(13 * k), SEMIBOLD, ALERT, "l"))
    for hand in phone.hands:
        p = hand.astype(np.int32)
        for a, b in HAND_LINKS:
            cv2.line(img, tuple(p[a]), tuple(p[b]), (210, 170, 90), 1, AA)
    draw_text(img, labels)


def _warning_icon(img: np.ndarray, cx: int, cy: int, size: int, color, ink) -> None:
    pts = np.array([(cx, cy - size), (cx + size, cy + int(size * 0.75)), (cx - size, cy + int(size * 0.75))], np.int32)
    cv2.fillPoly(img, [pts], ink, AA)
    cv2.line(img, (cx, cy - size // 3), (cx, cy + size // 5), color, max(2, size // 6), AA)
    cv2.circle(img, (cx, cy + int(size * 0.45)), max(1, size // 10), color, -1, AA)


def _detail(s: Status, headline: str) -> str | None:
    """A second banner line only when it says something the headline doesn't: how long, how far, how many."""
    for r in s.reasons:
        if r.startswith("Eyes closed"):
            return "Closed for " + r.removeprefix("Eyes closed ").replace("s", " s")
        if r.startswith("Eyes off road"):
            if abs(s.yaw) >= abs(s.pitch) and abs(s.yaw) > 0:
                return f"Head turned {abs(s.yaw):.0f}° away from the road"
            return f"Head down {abs(s.pitch):.0f}°" if s.pitch < 0 else "Eyes turned away"
        if r.startswith("Fatigue: PERCLOS"):
            return f"Eyes closed {r.rsplit(' ', 1)[-1]} of the last minute"
        if r.startswith("Fatigue:") and "yawns" in r:
            return f"{s.yawns} yawns in the last 5 minutes"
    return None


def _alert_overlay(img: np.ndarray, s: Status, t: float, k: float) -> None:
    """Flashing frame border plus a banner naming the reason, bottom centre, clear of the face."""
    h, w = img.shape[:2]
    color = LEVEL_COLOR[s.level]
    urgent = s.level == Level.ALERT
    ink = FG if urgent else (4, 20, 27)  # dark text on amber reads better than white
    # Border: ALERT flashes at ~2.5 Hz, WARN breathes slowly.
    phase = 0.5 + 0.5 * np.cos(2 * np.pi * t * (2.5 if urgent else 0.8))
    thick = round((10 if urgent else 6) * k)
    alpha = (0.3 + 0.55 * phase) if urgent else (0.25 + 0.35 * phase)
    for x0, y0, x1, y1 in ((0, 0, w, thick), (0, h - thick, w, h), (0, thick, thick, h - thick), (w - thick, thick, w, h - thick)):
        _blend_rect(img, x0, y0, x1, y1, color, alpha)

    cues = [c for c in (cue_for(r) for r in s.reasons) if c is not None]
    headline = cues[0].headline if cues else s.reasons[0].upper()
    detail = _detail(s, headline)
    hp, dp = round(30 * k), round(15 * k)
    icon, pad, gap = round(17 * k), round(22 * k), round(14 * k)
    hw = text_width(headline, hp, BOLD)
    dw = text_width(detail, dp, MEDIUM) if detail else 0
    bw = int(pad * 2 + icon * 2 + gap + max(hw, dw))
    bh = int(pad * 2 + hp * 0.74 + (dp + round(8 * k) if detail else 0))
    bx0, by0 = (w - bw) // 2, h - thick - round(28 * k) - bh
    _rounded_blend(img, bx0, by0, bx0 + bw, by0 + bh, round(16 * k), color, 0.94)
    tx = bx0 + pad + icon * 2 + gap
    base = by0 + pad + int(hp * 0.74)
    _warning_icon(img, bx0 + pad + icon, by0 + bh // 2, icon, color, ink)
    items = [(tx, base, headline, hp, BOLD, ink, "l")]
    if detail:
        items.append((tx, base + round(8 * k) + dp, detail, dp, MEDIUM, ink, "l"))
    draw_text(img, items)


def _panel(img: np.ndarray, face: FaceReading | None, s: Status, fps: float, phone: PhoneResult | None, cfg, k: float,
           phone_off: bool) -> None:
    h, w = img.shape[:2]
    margin, pad = round(20 * k), round(20 * k)
    pw = round(296 * k)
    row = round(46 * k)
    head = round(26 * k)
    foot = round(46 * k)
    ph = pad + head + round(14 * k) + 6 * row + foot + pad - round(12 * k)
    x0, y0 = w - pw - margin, margin
    x1, y1 = w - margin, min(h - margin, y0 + ph)
    _rounded_blend(img, x0, y0, x1, y1, round(14 * k), PANEL, 0.74)
    x, bw = x0 + pad, pw - 2 * pad
    texts = []

    learning = s.calibrating and s.level == Level.OK
    label = "Calibrating" if learning else LEVEL_LABEL[s.level]
    cy = y0 + pad + round(10 * k)
    cv2.circle(img, (x + round(5 * k), cy), round(5 * k), FG3 if learning else LEVEL_COLOR[s.level], -1, AA)
    texts.append((x + round(18 * k), cy + round(5 * k), label, round(15 * k), SEMIBOLD, FG, "l"))
    texts.append((x1 - pad, cy + round(5 * k), f"{fps:.0f} fps", round(12 * k), REGULAR, FG3, "r"))

    yl, pd = cfg.yaw_limit_deg, cfg.pitch_down_limit_deg
    pu, gl = cfg.pitch_up_limit_deg, cfg.gaze_limit
    readings = [  # same names, order and ranges as the web inspector
        ("ear", "Eye openness", face.ear if face else 0.0, 0.0, 0.45, [s.ear_threshold], ALERT if s.eyes_closed else GOOD, s.eyes_closed),
        ("jaw", "Jaw open", face.jaw_open if face else 0.0, 0.0, 1.0, [cfg.yawn_jaw], WARN if s.yawning else FG, s.yawning),
        ("yaw", "Head turn", s.yaw, -60, 60, [-yl, yl], FG, abs(s.yaw) > yl),
        ("pitch", "Head tilt", s.pitch, -45, 45, [-pd, pu], FG, s.pitch < -pd or s.pitch > pu),
        ("gaze", "Gaze", s.gaze, -1, 1, [-gl, gl], FG, abs(s.gaze) > gl),
        ("perclos", "PERCLOS, 60 s", s.perclos, 0.0, 0.4, [cfg.perclos_warn], FG, s.perclos >= cfg.perclos_warn),
    ]
    y = y0 + pad + head + round(14 * k)
    for key, name, value, lo, hi, marks, col, active in readings:
        base = y + round(11 * k)
        texts.append((x, base, name, round(13 * k), REGULAR, FG if active else FG2, "l"))
        texts.append((x + bw, base, _fmt(key, value), round(13 * k), MEDIUM, FG, "r"))
        _bar(img, x, y + round(24 * k), bw, k, value, lo, hi, marks, col if active or key in ("ear", "jaw") else FG2)
        y += row

    y += round(2 * k)
    cv2.line(img, (x, y), (x + bw, y), (46, 44, 44), 1, AA)
    base = y + round(26 * k)
    n_phones = len(phone.phones) if phone else 0
    texts.append((x, base, "Yawns, 5 min", round(13 * k), REGULAR, FG2, "l"))
    texts.append((x + round(98 * k), base, str(s.yawns), round(13 * k), MEDIUM, FG, "l"))
    state = "Detection off" if phone_off else "In view" if n_phones else "None"
    texts.append((x + bw, base, state, round(13 * k), MEDIUM, WARN if phone_off else ALERT if n_phones else FG2, "r"))
    texts.append((x + bw - text_width(state, round(13 * k), MEDIUM) - round(8 * k), base, "Phone", round(13 * k), REGULAR, FG2, "r"))
    draw_text(img, texts)


def _calibration_note(img: np.ndarray, s: Status, k: float) -> None:
    h = img.shape[0]
    msg = s.calibration_hint or "Hold your normal driving pose"
    px, pad = round(15 * k), round(16 * k)
    tw = text_width(msg, px, MEDIUM)
    bw, bh = int(pad * 2 + tw), round(52 * k)
    x0, y1 = round(20 * k), h - round(20 * k)
    _rounded_blend(img, x0, y1 - bh, x0 + bw, y1, round(12 * k), PANEL, 0.74)
    bar_y = y1 - round(12 * k)
    cv2.line(img, (x0 + pad, bar_y), (x0 + bw - pad, bar_y), TRACK, max(2, round(3 * k)), AA)
    fill = int((bw - 2 * pad) * float(np.clip(s.calibration_progress, 0, 1)))
    if fill > 0:
        cv2.line(img, (x0 + pad, bar_y), (x0 + pad + fill, bar_y), GOOD, max(2, round(3 * k)), AA)
    draw_text(img, [(x0 + pad, y1 - round(26 * k), msg, px, MEDIUM, FG, "l")])


def draw(img: np.ndarray, face: FaceReading | None, s: Status, fps: float,
         phone: PhoneResult | None = None, cfg=None, t: float = 0.0,
         phone_off: bool = False) -> np.ndarray:
    from .state import Config

    cfg = cfg or Config()
    k = float(np.clip(img.shape[0] / 720, 0.8, 2.5))

    if phone is not None:
        _detections(img, phone, k)
    if face is not None:
        eye_col = ALERT if s.eyes_closed else GOOD
        _contour(img, face.points, metrics.RIGHT_EYE_CONTOUR, eye_col)
        _contour(img, face.points, metrics.LEFT_EYE_CONTOUR, eye_col)
        _contour(img, face.points, metrics.MOUTH_CONTOUR, WARN if s.yawning else (200, 200, 200))
        for i in (metrics.RIGHT_IRIS, metrics.LEFT_IRIS):
            cv2.circle(img, tuple(face.points[i, :2].astype(int)), 2, FG, -1, AA)

    _panel(img, face, s, fps, phone, cfg, k, phone_off)
    if s.level > Level.OK and s.reasons:
        _alert_overlay(img, s, t, k)
    if s.calibrating and s.level == Level.OK:
        _calibration_note(img, s, k)
    return img

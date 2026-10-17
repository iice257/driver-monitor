"""Temporal reasoning: turns per-frame signals into driver states and alerts.

Frame-level signals are noisy (a single blink drops EAR to zero, detectors
flicker), so every decision here is based on how long a condition has held,
with a short grace period for dropouts, or on how often it happened inside a
sliding window. Time is passed in explicitly so the logic is deterministic and
testable against recorded or synthetic data.
"""

from __future__ import annotations

import tomllib
from collections import deque
from dataclasses import dataclass, field, fields
from enum import IntEnum
from pathlib import Path
from statistics import median


class Level(IntEnum):
    OK = 0
    WARN = 1
    ALERT = 2


@dataclass
class Config:
    calibration_s: float = 3.0
    smoothing: float = 0.35          # EMA factor for pose and gaze (1 = no smoothing)

    # Eyes
    ear_ratio: float = 0.72          # closed when EAR < baseline * ratio
    ear_floor: float = 0.14
    blink_score: float = 0.65        # or when the eyeBlink blendshape exceeds this
    microsleep_s: float = 1.2
    perclos_window_s: float = 60.0
    perclos_warn: float = 0.15

    # Yawning
    yawn_jaw: float = 0.45           # jawOpen blendshape
    yawn_mar: float = 0.45           # fallback when blendshapes are unavailable
    yawn_min_s: float = 1.0
    yawn_notice_s: float = 3.0       # show "Yawn" briefly after each one
    yawn_window_s: float = 300.0
    yawn_warn_count: int = 3

    # Attention
    yaw_limit_deg: float = 25.0
    pitch_down_limit_deg: float = 15.0
    pitch_up_limit_deg: float = 25.0
    gaze_limit: float = 0.45         # iris offset, -1..1
    hysteresis: float = 0.75         # leave the "away" state below limit * this
    distraction_s: float = 2.0
    no_face_s: float = 2.0

    # Phone
    phone_s: float = 1.0
    phone_down_s: float = 1.0        # phone visible + looking down
    hand_at_ear_s: float = 2.0
    phone_grace_s: float = 0.6       # tolerate detector dropouts this long

    @classmethod
    def load(cls, path: Path | None) -> "Config":
        if path is None or not Path(path).exists():
            return cls()
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        flat = {k: v for section in data.values() if isinstance(section, dict) for k, v in section.items()}
        flat.update({k: v for k, v in data.items() if not isinstance(v, dict)})
        known = {f.name for f in fields(cls)}
        unknown = set(flat) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        return cls(**flat)


@dataclass
class Frame:
    t: float
    face: bool
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gaze: float = 0.0
    blink: float | None = None       # blendshape score, if available
    jaw_open: float | None = None
    phone: bool = False              # phone detected anywhere in view
    phone_near_face: bool = False
    hand_at_ear: bool = False


@dataclass
class Status:
    level: Level
    reasons: list[str]
    calibrating: bool
    eyes_closed: bool
    ear_threshold: float
    perclos: float
    yawns: int
    closed_for: float
    away_for: float
    yaw: float = 0.0                 # smoothed, relative to calibrated baseline
    pitch: float = 0.0
    gaze: float = 0.0
    yawning: bool = False


@dataclass
class _Timer:
    """How long a condition has held, ignoring dropouts shorter than `grace`."""

    grace: float = 0.0
    since: float | None = None
    last_true: float | None = None

    def update(self, active: bool, t: float) -> float:
        if active:
            if self.since is None:
                self.since = t
            self.last_true = t
        elif self.since is not None and (self.last_true is None or t - self.last_true > self.grace):
            self.since = self.last_true = None
        return 0.0 if self.since is None else t - self.since


class _Ema:
    def __init__(self, alpha: float) -> None:
        self.alpha, self.value = alpha, None

    def __call__(self, x: float) -> float:
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        return self.value


@dataclass
class DriverState:
    cfg: Config = field(default_factory=Config)

    def __post_init__(self) -> None:
        cfg = self.cfg
        self._start: float | None = None
        self._calib: list[Frame] = []
        self.baseline_ear: float | None = None
        self.baseline_pitch = self.baseline_yaw = self.baseline_gaze = 0.0
        self._closed = _Timer()
        self._away = _Timer(grace=0.3)
        self._mouth = _Timer(grace=0.2)
        self._missing = _Timer()
        self._phone = _Timer(grace=cfg.phone_grace_s)
        self._phone_down = _Timer(grace=cfg.phone_grace_s)
        self._ear_hand = _Timer(grace=cfg.phone_grace_s)
        self._yaw, self._pitch, self._gaze = _Ema(cfg.smoothing), _Ema(cfg.smoothing), _Ema(cfg.smoothing)
        self._is_away = False
        self._yawn_counted = False
        self._closed_samples: deque[tuple[float, bool]] = deque()
        self._yawns: deque[float] = deque()

    @property
    def calibrated(self) -> bool:
        return self.baseline_ear is not None

    def threshold(self) -> float:
        if self.baseline_ear is None:
            return 0.21  # textbook default until we know this face
        return max(self.cfg.ear_floor, self.baseline_ear * self.cfg.ear_ratio)

    def recalibrate(self) -> None:
        self.__post_init__()

    def _calibrate(self, f: Frame) -> bool:
        if self._start is None:
            self._start = f.t
        if f.face:
            self._calib.append(f)
        if f.t - self._start < self.cfg.calibration_s or len(self._calib) < 10:
            return True
        # Upper half of EAR samples, so blinks during calibration don't drag it down.
        ears = sorted(c.ear for c in self._calib)
        self.baseline_ear = median(ears[len(ears) // 2:])
        self.baseline_pitch = median(c.pitch for c in self._calib)
        self.baseline_yaw = median(c.yaw for c in self._calib)
        self.baseline_gaze = median(c.gaze for c in self._calib)
        return False

    def _perclos(self, t: float, closed: bool) -> float:
        self._closed_samples.append((t, closed))
        while self._closed_samples and t - self._closed_samples[0][0] > self.cfg.perclos_window_s:
            self._closed_samples.popleft()
        # Time-weighted so a dropped frame rate doesn't skew the ratio.
        total = closed_time = 0.0
        samples = list(self._closed_samples)
        for (t0, c0), (t1, _) in zip(samples, samples[1:]):
            total += t1 - t0
            if c0:
                closed_time += t1 - t0
        return closed_time / total if total > 0 else 0.0

    def _attention(self, f: Frame, calibrating: bool) -> tuple[float, float, float, float]:
        cfg = self.cfg
        if not f.face:
            return 0.0, 0.0, 0.0, self._away.update(False, f.t)
        d_yaw = self._yaw(f.yaw) - self.baseline_yaw
        d_pitch = self._pitch(f.pitch) - self.baseline_pitch
        d_gaze = self._gaze(f.gaze) - self.baseline_gaze
        if calibrating:
            return d_yaw, d_pitch, d_gaze, 0.0
        # Hysteresis: once away, the driver must come clearly back to clear it.
        k = cfg.hysteresis if self._is_away else 1.0
        self._is_away = (
            abs(d_yaw) > cfg.yaw_limit_deg * k
            or d_pitch < -cfg.pitch_down_limit_deg * k
            or d_pitch > cfg.pitch_up_limit_deg * k
            or abs(d_gaze) > cfg.gaze_limit * k
        )
        return d_yaw, d_pitch, d_gaze, self._away.update(self._is_away, f.t)

    def update(self, f: Frame) -> Status:
        cfg = self.cfg
        calibrating = not self.calibrated and self._calibrate(f)
        missing_for = self._missing.update(not f.face, f.t)

        # Eyes: EAR against the personal threshold, or a confident blink blendshape.
        thr = self.threshold()
        closed = f.face and (f.ear < thr or (f.blink is not None and f.blink > cfg.blink_score))
        closed_for = self._closed.update(closed, f.t)
        perclos = self._perclos(f.t, closed)

        # Yawns: jaw held open, which talking or a quick laugh doesn't do.
        mouth_open = f.face and (
            f.jaw_open > cfg.yawn_jaw if f.jaw_open is not None else f.mar > cfg.yawn_mar
        )
        open_for = self._mouth.update(mouth_open, f.t)
        yawning = open_for >= cfg.yawn_min_s
        if yawning and not self._yawn_counted:
            self._yawns.append(f.t)
            self._yawn_counted = True
        elif open_for == 0.0:
            self._yawn_counted = False
        while self._yawns and f.t - self._yawns[0] > cfg.yawn_window_s:
            self._yawns.popleft()
        recent_yawn = bool(self._yawns) and f.t - self._yawns[-1] < cfg.yawn_notice_s

        d_yaw, d_pitch, d_gaze, away_for = self._attention(f, calibrating)

        phone_for = self._phone.update(f.phone_near_face, f.t)
        phone_down_for = self._phone_down.update(f.phone and d_pitch < -cfg.pitch_down_limit_deg * 0.6, f.t)
        ear_hand_for = self._ear_hand.update(f.hand_at_ear, f.t)

        level, reasons = Level.OK, []

        def raise_(lvl: Level, why: str) -> None:
            nonlocal level
            level = max(level, lvl)
            reasons.append(why)

        if not calibrating:
            if closed_for >= cfg.microsleep_s:
                raise_(Level.ALERT, f"Eyes closed {closed_for:.1f}s")
            if phone_for >= cfg.phone_s or phone_down_for >= cfg.phone_down_s:
                raise_(Level.ALERT, "Phone in use")
            elif ear_hand_for >= cfg.hand_at_ear_s:
                raise_(Level.ALERT, "Phone call")
            if away_for >= cfg.distraction_s and "Phone in use" not in reasons:
                raise_(Level.ALERT, "Eyes off road")
            if missing_for >= cfg.no_face_s:
                raise_(Level.WARN, "Driver not visible")
            if perclos >= cfg.perclos_warn:
                raise_(Level.WARN, f"Fatigue: PERCLOS {perclos:.0%}")
            if len(self._yawns) >= cfg.yawn_warn_count:
                raise_(Level.WARN, f"Fatigue: {len(self._yawns)} yawns")
            elif recent_yawn:
                raise_(Level.WARN, "Yawn")

        return Status(
            level=level,
            reasons=reasons,
            calibrating=calibrating,
            eyes_closed=closed,
            ear_threshold=thr,
            perclos=perclos,
            yawns=len(self._yawns),
            closed_for=closed_for,
            away_for=away_for,
            yaw=d_yaw,
            pitch=d_pitch,
            gaze=d_gaze,
            yawning=yawning,
        )

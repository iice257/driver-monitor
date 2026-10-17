"""Temporal reasoning: turns per-frame signals into driver states and alerts.

Frame-level signals are noisy (a single blink drops EAR to zero), so every
decision here is based on how long a condition has held, or on how often it
happened inside a sliding window. Time is passed in explicitly so the logic is
deterministic and testable against recorded or synthetic data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from statistics import median


class Level(IntEnum):
    OK = 0
    WARN = 1
    ALERT = 2


@dataclass
class Config:
    calibration_s: float = 3.0
    ear_ratio: float = 0.72          # closed when EAR < baseline * ratio
    ear_floor: float = 0.14          # never let the threshold drop below this
    microsleep_s: float = 1.2        # eyes closed this long -> ALERT
    perclos_window_s: float = 60.0
    perclos_warn: float = 0.15       # >15% of the window with eyes closed -> WARN
    yawn_mar: float = 0.55
    yawn_min_s: float = 0.8
    yawn_window_s: float = 300.0
    yawn_warn_count: int = 3
    yaw_limit_deg: float = 28.0
    pitch_down_limit_deg: float = 18.0
    distraction_s: float = 2.0
    no_face_s: float = 2.0


@dataclass
class Frame:
    t: float
    face: bool
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


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


@dataclass
class _Timer:
    """Tracks how long a boolean condition has been continuously true."""

    since: float | None = None

    def update(self, active: bool, t: float) -> float:
        if not active:
            self.since = None
            return 0.0
        if self.since is None:
            self.since = t
        return t - self.since


@dataclass
class DriverState:
    cfg: Config = field(default_factory=Config)

    def __post_init__(self) -> None:
        self._start: float | None = None
        self._calib_ear: list[float] = []
        self._calib_pose: list[tuple[float, float]] = []
        self.baseline_ear: float | None = None
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self._closed = _Timer()
        self._away = _Timer()
        self._mouth = _Timer()
        self._missing = _Timer()
        self._yawn_counted = False
        self._closed_samples: deque[tuple[float, bool]] = deque()
        self._yawns: deque[float] = deque()
        self._last_t: float | None = None

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
            self._calib_ear.append(f.ear)
            self._calib_pose.append((f.pitch, f.yaw))
        if f.t - self._start < self.cfg.calibration_s or len(self._calib_ear) < 10:
            return True
        # Upper half of the samples, so blinks during calibration don't drag it down.
        ears = sorted(self._calib_ear)
        self.baseline_ear = median(ears[len(ears) // 2:])
        self.baseline_pitch = median(p for p, _ in self._calib_pose)
        self.baseline_yaw = median(y for _, y in self._calib_pose)
        return False

    def _perclos(self, t: float, closed: bool) -> float:
        cfg = self.cfg
        self._closed_samples.append((t, closed))
        while self._closed_samples and t - self._closed_samples[0][0] > cfg.perclos_window_s:
            self._closed_samples.popleft()
        # Time-weighted so a dropped frame rate doesn't skew the ratio.
        total = closed_time = 0.0
        samples = list(self._closed_samples)
        for (t0, c0), (t1, _) in zip(samples, samples[1:]):
            dt = t1 - t0
            total += dt
            if c0:
                closed_time += dt
        return closed_time / total if total > 0 else 0.0

    def update(self, f: Frame) -> Status:
        cfg = self.cfg
        calibrating = not self.calibrated and self._calibrate(f)
        missing_for = self._missing.update(not f.face, f.t)

        thr = self.threshold()
        closed = f.face and f.ear < thr
        closed_for = self._closed.update(closed, f.t)
        perclos = self._perclos(f.t, closed) if f.face else self._perclos(f.t, False)

        open_for = self._mouth.update(f.face and f.mar > cfg.yawn_mar, f.t)
        if open_for >= cfg.yawn_min_s and not self._yawn_counted:
            self._yawns.append(f.t)
            self._yawn_counted = True
        elif open_for == 0.0:
            self._yawn_counted = False
        while self._yawns and f.t - self._yawns[0] > cfg.yawn_window_s:
            self._yawns.popleft()

        d_yaw = f.yaw - self.baseline_yaw
        d_pitch = f.pitch - self.baseline_pitch
        looking_away = f.face and not calibrating and (
            abs(d_yaw) > cfg.yaw_limit_deg or d_pitch < -cfg.pitch_down_limit_deg
        )
        away_for = self._away.update(looking_away, f.t)

        level, reasons = Level.OK, []

        def raise_(lvl: Level, why: str) -> None:
            nonlocal level
            level = max(level, lvl)
            reasons.append(why)

        if not calibrating:
            if closed_for >= cfg.microsleep_s:
                raise_(Level.ALERT, f"Eyes closed {closed_for:.1f}s")
            if away_for >= cfg.distraction_s:
                raise_(Level.ALERT, "Eyes off road")
            if missing_for >= cfg.no_face_s:
                raise_(Level.WARN, "Driver not visible")
            if perclos >= cfg.perclos_warn:
                raise_(Level.WARN, f"Fatigue: PERCLOS {perclos:.0%}")
            if len(self._yawns) >= cfg.yawn_warn_count:
                raise_(Level.WARN, f"Fatigue: {len(self._yawns)} yawns")

        self._last_t = f.t
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
        )

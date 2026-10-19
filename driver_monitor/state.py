"""Temporal reasoning: turns per-frame signals into driver states and alerts.

Frame-level signals are noisy (a single blink drops EAR to zero, detectors
flicker), so every decision here is based on how long a condition has held,
with a short grace period for dropouts, or on how often it happened inside a
sliding window. Time is passed in explicitly so the logic is deterministic and
testable against recorded or synthetic data.
"""

from __future__ import annotations

import math
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



def _half_up(x: float) -> int:
    """Round half up, with a hair of tolerance so Python and the JS port always print the same digit."""
    return math.floor(x + 0.5 + 1e-6)


@dataclass
class Config:
    smoothing: float = 0.35          # EMA factor for pose and gaze (1 = no smoothing)

    # Calibration: the baseline is learned only from frames that show an attentive driver.
    calib_good_s: float = 2.0        # seconds of good frames needed
    calib_window_s: float = 8.0      # ... within this rolling window
    calib_ear_min: float = 0.2       # eyes clearly open (EAR above this, or blink score below calib_blink_max)
    calib_blink_max: float = 0.3
    calib_jaw_max: float = 0.2       # mouth closed
    calib_pose_deg: float = 15.0     # head near frontal, |yaw| and |pitch| below this

    # Eyes
    ear_ratio: float = 0.72          # closed when EAR < baseline * ratio
    ear_floor: float = 0.14
    blink_score: float = 0.65        # or when the eyeBlink blendshape exceeds this
    microsleep_s: float = 1.2
    perclos_window_s: float = 60.0
    perclos_warn: float = 0.15
    perclos_min_s: float = 20.0      # don't judge PERCLOS on less history than this

    # Yawning
    yawn_jaw: float = 0.45           # jawOpen blendshape
    yawn_mar: float = 0.45           # fallback when blendshapes are unavailable
    yawn_min_s: float = 1.0
    yawn_notice_s: float = 3.0       # show "Yawn" briefly after each one
    yawn_window_s: float = 300.0
    yawn_warn_count: int = 3
    yawn_squint_s: float = 2.0       # eyes squeezed shut during a real yawn this long aren't a microsleep
    eye_pose_yaw_deg: float = 30.0   # beyond this head turn EAR is unreliable: trust the blink blendshape only
    eye_pose_pitch_deg: float = 25.0 # ... and beyond this much head-down

    # Attention
    yaw_limit_deg: float = 25.0
    pitch_down_limit_deg: float = 15.0
    pitch_up_limit_deg: float = 25.0
    gaze_limit: float = 0.45         # iris offset, -1..1
    hysteresis: float = 0.75         # leave the "away" state below limit * this
    distraction_s: float = 2.0
    no_face_s: float = 2.0           # driver not visible: WARN ...
    no_face_alert_s: float = 5.0     # ... then ALERT

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
    calibration_progress: float = 1.0
    calibration_hint: str | None = None  # why the last frame didn't count towards calibration


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
        self._calib_last: float | None = None
        self._calib: deque[tuple[float, float, Frame]] = deque()  # good frames: (t, dt, frame)
        self._calib_good = 0.0
        self._calib_hint: str | None = None
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

    def _calib_problem(self, f: Frame) -> str | None:
        """Why this frame can't teach us what an attentive driver looks like, or None if it can."""
        cfg = self.cfg
        if not f.face:
            return "Face the camera"
        if not (f.ear > cfg.calib_ear_min or (f.blink is not None and f.blink < cfg.calib_blink_max)):
            return "Open your eyes"
        jaw_open = f.jaw_open > cfg.calib_jaw_max if f.jaw_open is not None else f.mar > cfg.yawn_mar
        if jaw_open:
            return "Close your mouth"
        if abs(f.yaw) > cfg.calib_pose_deg or abs(f.pitch) > cfg.calib_pose_deg:
            return "Look at the road"
        return None

    def _calibrate(self, f: Frame) -> bool:
        cfg = self.cfg
        dt = 0.0 if self._calib_last is None else min(max(f.t - self._calib_last, 0.0), 0.25)
        self._calib_last = f.t
        self._calib_hint = self._calib_problem(f)
        if self._calib_hint is None:
            self._calib.append((f.t, dt, f))
        while self._calib and f.t - self._calib[0][0] > cfg.calib_window_s:
            self._calib.popleft()
        self._calib_good = sum(dt for _, dt, _ in self._calib)
        if self._calib_good < cfg.calib_good_s or len(self._calib) < 10:
            return True
        frames = [c[2] for c in self._calib]
        # Upper half of EAR samples, so blinks during calibration don't drag it down.
        ears = sorted(c.ear for c in frames)
        self.baseline_ear = median(ears[len(ears) // 2:])
        self.baseline_pitch = median(c.pitch for c in frames)
        self.baseline_yaw = median(c.yaw for c in frames)
        self.baseline_gaze = median(c.gaze for c in frames)
        self._calib_hint = None
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
        if total < min(self.cfg.perclos_min_s, self.cfg.perclos_window_s):
            return 0.0  # a few seconds of history makes one blink look like fatigue
        return closed_time / total

    def _attention(self, f: Frame) -> tuple[float, float, float, float]:
        cfg = self.cfg
        if not f.face:
            return 0.0, 0.0, 0.0, self._away.update(False, f.t)
        d_yaw = self._yaw(f.yaw) - self.baseline_yaw
        d_pitch = self._pitch(f.pitch) - self.baseline_pitch
        d_gaze = self._gaze(f.gaze) - self.baseline_gaze
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

        # Yawns: jaw held open, which talking or a quick laugh doesn't do.
        jaw = f.jaw_open
        mouth_open = f.face and (jaw > cfg.yawn_jaw if jaw is not None else f.mar > cfg.yawn_mar)
        open_for = self._mouth.update(mouth_open, f.t)
        # People squeeze their eyes shut while yawning; that isn't a microsleep. Only a real yawn
        # (jaw past the yawn threshold) earns the exemption, and only for yawn_squint_s: a slack
        # jaw with shut eyes is a driver falling asleep.
        squint = mouth_open and open_for < cfg.yawn_squint_s

        # Eyes: EAR against the personal threshold, or a confident blink blendshape. With the head
        # turned or tipped far from the camera EAR is unreliable, so only the blendshape is used.
        thr = self.threshold()
        blink_closed = f.blink is not None and f.blink > cfg.blink_score
        if abs(f.yaw) > cfg.eye_pose_yaw_deg or f.pitch < -cfg.eye_pose_pitch_deg:
            shut = blink_closed
        else:
            shut = f.ear < thr or blink_closed
        closed = f.face and not squint and shut
        closed_for = self._closed.update(closed, f.t)
        perclos = self._perclos(f.t, closed)
        yawning = open_for >= cfg.yawn_min_s
        if yawning and not self._yawn_counted:
            self._yawns.append(f.t)
            self._yawn_counted = True
        elif open_for == 0.0:
            self._yawn_counted = False
        while self._yawns and f.t - self._yawns[0] > cfg.yawn_window_s:
            self._yawns.popleft()
        recent_yawn = bool(self._yawns) and f.t - self._yawns[-1] < cfg.yawn_notice_s

        d_yaw, d_pitch, d_gaze, away_for = self._attention(f)

        phone_for = self._phone.update(f.phone_near_face, f.t)
        phone_down_for = self._phone_down.update(f.phone and d_pitch < -cfg.pitch_down_limit_deg * 0.6, f.t)
        ear_hand_for = self._ear_hand.update(f.hand_at_ear, f.t)

        level, reasons = Level.OK, []

        def raise_(lvl: Level, why: str) -> None:
            nonlocal level
            level = max(level, lvl)
            reasons.append(why)

        # Alerts are live from the first frame: until the baseline is learned they run on the
        # textbook thresholds (EAR 0.21, pose relative to the camera), so a driver who is already
        # asleep when the monitor starts is not ignored.
        if closed_for >= cfg.microsleep_s:
            raise_(Level.ALERT, f"Eyes closed {_half_up(closed_for * 10) / 10:.1f}s")
        if phone_for >= cfg.phone_s or phone_down_for >= cfg.phone_down_s:
            raise_(Level.ALERT, "Phone in use")
        elif ear_hand_for >= cfg.hand_at_ear_s:
            raise_(Level.ALERT, "Phone call")
        if away_for >= cfg.distraction_s and "Phone in use" not in reasons:
            raise_(Level.ALERT, "Eyes off road")
        if missing_for >= cfg.no_face_alert_s:
            raise_(Level.ALERT, "Driver not visible")
        elif missing_for >= cfg.no_face_s:
            raise_(Level.WARN, "Driver not visible")
        if perclos >= cfg.perclos_warn:
            raise_(Level.WARN, f"Fatigue: PERCLOS {_half_up(perclos * 100)}%")
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
            calibration_progress=1.0 if not calibrating else min(1.0, self._calib_good / cfg.calib_good_s),
            calibration_hint=self._calib_hint if calibrating else None,
        )

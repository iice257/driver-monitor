import math

import numpy as np

from driver_monitor import metrics
from driver_monitor.state import Config, DriverState, Frame, Level

FPS = 30
OPEN, CLOSED = 0.30, 0.08


def run(state, seconds, start=0.0, **signals):
    s = None
    for i in range(int(seconds * FPS)):
        s = state.update(Frame(t=start + i / FPS, face=signals.get("face", True),
                               ear=signals.get("ear", OPEN), mar=signals.get("mar", 0.2),
                               pitch=signals.get("pitch", 0.0), yaw=signals.get("yaw", 0.0)))
    return s, start + seconds


def calibrated():
    st = DriverState(Config())
    s, t = run(st, 3.5)
    assert not s.calibrating
    return st, t


def test_calibration_sets_threshold_from_baseline():
    st, _ = calibrated()
    assert math.isclose(st.baseline_ear, OPEN)
    assert math.isclose(st.threshold(), OPEN * 0.72)


def test_blink_does_not_alert():
    st, t = calibrated()
    s, t = run(st, 0.2, t, ear=CLOSED)
    assert s.eyes_closed and s.level == Level.OK


def test_microsleep_alerts():
    st, t = calibrated()
    s, t = run(st, 1.5, t, ear=CLOSED)
    assert s.level == Level.ALERT
    s, t = run(st, 0.1, t)
    assert s.level < Level.ALERT


def test_looking_away_alerts_after_delay():
    st, t = calibrated()
    s, t = run(st, 1.0, t, yaw=40)
    assert s.level == Level.OK
    s, t = run(st, 1.5, t, yaw=40)
    assert s.level == Level.ALERT and "Eyes off road" in s.reasons


def test_three_yawns_warn():
    st, t = calibrated()
    for _ in range(3):
        s, t = run(st, 1.2, t, mar=0.8)
        s, t = run(st, 2.0, t)
    assert s.yawns == 3 and s.level == Level.WARN


def test_missing_face_warns():
    st, t = calibrated()
    s, t = run(st, 2.5, t, face=False)
    assert s.level == Level.WARN and "Driver not visible" in s.reasons


def test_ear_on_synthetic_eye():
    pts = np.zeros((478, 2))
    # Right eye: width 30px, lids 6px apart -> EAR 0.2
    for i, p in zip(metrics.RIGHT_EYE, [(0, 0), (10, -3), (20, -3), (30, 0), (20, 3), (10, 3)]):
        pts[i] = p
    assert math.isclose(metrics.eye_aspect_ratio(pts, metrics.RIGHT_EYE), 0.2)


def test_euler_identity_and_yaw():
    assert metrics.euler_from_matrix(np.eye(4)) == (0.0, 0.0, 0.0)
    a = math.radians(30)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    _, yaw, _ = metrics.euler_from_matrix(ry)
    assert math.isclose(yaw, 30.0, abs_tol=1e-6)

import math
from pathlib import Path

import numpy as np
import pytest

from driver_monitor import metrics
from driver_monitor.phone import hand_at_ear, phone_near_face
from driver_monitor.state import Config, DriverState, Frame, Level

FPS = 30
OPEN, CLOSED = 0.30, 0.08


def run(state, seconds, start=0.0, **signals):
    s = None
    defaults = dict(face=True, ear=OPEN, mar=0.2, pitch=0.0, yaw=0.0, gaze=0.0, blink=0.05, jaw_open=0.05)
    defaults.update(signals)
    for i in range(int(seconds * FPS)):
        s = state.update(Frame(t=start + i / FPS, **defaults))
    return s, start + seconds


def calibrated(**cfg):
    st = DriverState(Config(**cfg))
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


def test_blink_blendshape_alone_counts_as_closed():
    st, t = calibrated()
    s, t = run(st, 1.5, t, blink=0.9)
    assert s.eyes_closed and s.level == Level.ALERT


def test_looking_away_alerts_after_delay():
    st, t = calibrated()
    s, t = run(st, 1.0, t, yaw=40)
    assert s.level == Level.OK
    s, t = run(st, 1.5, t, yaw=40)
    assert s.level == Level.ALERT and "Eyes off road" in s.reasons


def test_quick_mirror_check_is_fine():
    st, t = calibrated()
    s, t = run(st, 1.2, t, yaw=40)
    s, t = run(st, 2.0, t)
    assert s.level == Level.OK


def test_eyes_only_glance_alerts():
    st, t = calibrated()
    s, t = run(st, 2.5, t, gaze=0.8)
    assert "Eyes off road" in s.reasons


def test_looking_down_alerts_relative_to_baseline():
    st = DriverState(Config())
    _, t = run(st, 3.5, pitch=-10)  # camera mounted high: baseline is already looking down
    s, t = run(st, 2.5, t, pitch=-15)
    assert s.level == Level.OK
    s, t = run(st, 2.5, t, pitch=-35)
    assert "Eyes off road" in s.reasons


def test_single_yawn_is_reported_and_talking_is_not():
    st, t = calibrated()
    for _ in range(6):  # talking: jaw opens briefly
        s, t = run(st, 0.3, t, jaw_open=0.6)
        s, t = run(st, 0.2, t)
    assert s.yawns == 0
    s, t = run(st, 1.3, t, jaw_open=0.7)
    assert s.yawns == 1 and "Yawn" in s.reasons


def test_three_yawns_warn():
    st, t = calibrated()
    for _ in range(3):
        s, t = run(st, 1.3, t, jaw_open=0.7)
        s, t = run(st, 2.0, t)
    assert s.yawns == 3 and s.level == Level.WARN


def test_phone_near_face_alerts_through_flicker():
    st, t = calibrated()
    for _ in range(6):  # detector sees the phone 2 frames in 3
        s, t = run(st, 0.2, t, phone_near_face=True)
        s, t = run(st, 0.1, t)
    assert "Phone in use" in s.reasons and s.level == Level.ALERT


def test_phone_while_looking_down_alerts():
    st, t = calibrated()
    s, t = run(st, 1.3, t, phone=True, pitch=-20)
    assert "Phone in use" in s.reasons


def test_hand_at_ear_is_a_call():
    st, t = calibrated()
    s, t = run(st, 2.5, t, hand_at_ear=True)
    assert "Phone call" in s.reasons


def test_missing_face_warns():
    st, t = calibrated()
    s, t = run(st, 2.5, t, face=False)
    assert s.level == Level.WARN and "Driver not visible" in s.reasons


def test_config_file_loads_sections(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text("[yawn]\nyawn_jaw = 0.3\n[eyes]\nmicrosleep_s = 2.0\n")
    cfg = Config.load(p)
    assert cfg.yawn_jaw == 0.3 and cfg.microsleep_s == 2.0
    p.write_text("[eyes]\ntypo = 1\n")
    with pytest.raises(ValueError):
        Config.load(p)


def test_repo_config_matches_defaults():
    assert Config.load(Path(__file__).parent.parent / "config.toml") == Config()


# --- geometry -------------------------------------------------------------

def _face(yaw_deg=0.0, pitch_deg=0.0):
    """Synthetic 3D face: eye corners and forehead/chin rotated, camera looks along +z."""
    pts = np.zeros((478, 3))
    pts[33], pts[263] = (-40, 0, 0), (40, 0, 0)
    pts[10], pts[152] = (0, -60, 0), (0, 60, 0)
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    # yaw about the vertical axis, then pitch about the horizontal axis
    ry = np.array([[math.cos(y), 0, -math.sin(y)], [0, 1, 0], [math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    return pts @ (rx @ ry).T


def test_head_pose_frontal():
    pitch, yaw, roll = metrics.head_pose(_face())
    assert abs(pitch) < 1e-6 and abs(yaw) < 1e-6 and abs(roll) < 1e-6


@pytest.mark.parametrize("deg", [-50, -20, 20, 50])
def test_head_pose_yaw(deg):
    _, yaw, _ = metrics.head_pose(_face(yaw_deg=deg))
    assert math.isclose(yaw, deg, abs_tol=0.5)


@pytest.mark.parametrize("deg", [-30, 30])
def test_head_pose_pitch_no_wraparound(deg):
    pitch, _, _ = metrics.head_pose(_face(pitch_deg=deg))
    assert math.isclose(pitch, deg, abs_tol=0.5)


def test_gaze_ratio():
    pts = np.zeros((478, 3))
    pts[33], pts[133], pts[362], pts[263] = (0, 0, 0), (30, 0, 0), (60, 0, 0), (90, 0, 0)
    pts[468], pts[473] = (15, 0, 0), (75, 0, 0)
    assert math.isclose(metrics.gaze_ratio(pts), 0.0)
    pts[468], pts[473] = (22.5, 0, 0), (82.5, 0, 0)
    assert math.isclose(metrics.gaze_ratio(pts), 0.5)


def test_ear_on_synthetic_eye():
    pts = np.zeros((478, 2))
    for i, p in zip(metrics.RIGHT_EYE, [(0, 0), (10, -3), (20, -3), (30, 0), (20, 3), (10, 3)]):
        pts[i] = p
    assert math.isclose(metrics.eye_aspect_ratio(pts, metrics.RIGHT_EYE), 0.2)


def test_phone_geometry():
    face = (200, 100, 300, 220)
    hand = np.tile([175, 160], (21, 1)).astype(float)
    assert hand_at_ear(hand, face)
    assert not hand_at_ear(np.tile([250, 400], (21, 1)).astype(float), face)
    assert phone_near_face([((210, 200, 260, 280), 0.6)], face)
    assert not phone_near_face([((600, 400, 650, 470), 0.6)], face)


def test_squint_during_yawn_is_not_a_microsleep():
    st, t = calibrated()
    s, t = run(st, 2.0, t, jaw_open=0.7, ear=CLOSED)
    assert "Yawn" in s.reasons and s.level < Level.ALERT


def test_eyes_closed_with_mouth_hanging_open_still_alerts():
    st, t = calibrated()
    s, t = run(st, 8.0, t, jaw_open=0.7, ear=CLOSED)
    assert any(r.startswith("Eyes closed") for r in s.reasons)


def test_perclos_needs_history():
    st, t = calibrated()
    s, t = run(st, 3.0, t, ear=CLOSED)  # mostly closed, but only a few seconds of history
    assert not any(r.startswith("Fatigue: PERCLOS") for r in s.reasons)


# --- calibration validation, squint cap, pose gating, no-face escalation -----

def _calibrate_through(st, bad_s, **bad):
    """A bad stretch first, then a clean one; returns the status after calibration finished."""
    s, t = run(st, bad_s, **bad)
    assert s.calibrating, "bad frames must not complete calibration"
    s, t = run(st, 3.0, t)
    return s, t


def test_calibrating_through_a_yawn_does_not_poison_the_baseline():
    st = DriverState(Config())
    s, t = _calibrate_through(st, 3.0, jaw_open=0.8, mar=0.9)
    assert not s.calibrating
    assert st.baseline_ear == OPEN and abs(st.baseline_yaw) < 1e-9


def test_calibrating_with_eyes_closed_does_not_poison_the_baseline():
    st = DriverState(Config())
    s, t = _calibrate_through(st, 3.0, ear=CLOSED, blink=0.9)
    assert st.baseline_ear == OPEN and st.threshold() == OPEN * 0.72


def test_calibrating_with_the_head_turned_does_not_poison_the_baseline():
    st = DriverState(Config())
    s, t = _calibrate_through(st, 3.0, yaw=40, pitch=-20)
    assert abs(st.baseline_yaw) < 1e-9 and abs(st.baseline_pitch) < 1e-9


def test_calibration_says_what_is_wrong():
    for signals, hint in [
        (dict(face=False), "Face the camera"),
        (dict(ear=CLOSED, blink=0.9), "Open your eyes"),
        (dict(jaw_open=0.7), "Close your mouth"),
        (dict(yaw=30), "Look at the road"),
    ]:
        s, _ = run(DriverState(Config()), 1.0, **signals)
        assert s.calibrating and s.calibration_hint == hint


def test_calibration_needs_good_time_inside_a_rolling_window():
    st = DriverState(Config())
    t = 0.0
    for _ in range(4):  # 1 s good, 1.5 s bad, repeated: never 2 s of good inside 8 s
        s, t = run(st, 0.4, t)
        s, t = run(st, 2.5, t, yaw=40)
    assert s.calibrating
    s, t = run(st, 3.0, t)
    assert not s.calibrating


def test_microsleep_during_calibration_still_alerts_on_the_textbook_threshold():
    st = DriverState(Config())
    s, t = run(st, 0.5)
    s, t = run(st, 2.0, t, ear=CLOSED, blink=0.9)
    assert s.calibrating and s.level == Level.ALERT and any(r.startswith("Eyes closed") for r in s.reasons)


def test_slack_jaw_is_not_a_yawn_squint():
    st, t = calibrated()
    s, t = run(st, 2.0, t, jaw_open=0.35, ear=CLOSED, blink=0.1)
    assert s.level == Level.ALERT and any(r.startswith("Eyes closed") for r in s.reasons)


def test_yawn_squint_exemption_is_capped():
    st, t = calibrated()
    s, t = run(st, 2.0, t, jaw_open=0.7, ear=CLOSED)
    assert not any(r.startswith("Eyes closed") for r in s.reasons)
    s, t = run(st, 2.0, t, jaw_open=0.7, ear=CLOSED)
    assert any(r.startswith("Eyes closed") for r in s.reasons)


def test_extreme_head_pose_judges_closure_from_the_blink_blendshape_only():
    st, t = calibrated()
    s, t = run(st, 2.0, t, yaw=-60, ear=0.77, blink=0.9)
    assert s.eyes_closed and any(r.startswith("Eyes closed") for r in s.reasons)
    st, t = calibrated()
    s, t = run(st, 2.0, t, yaw=-60, ear=0.05, blink=0.1)  # EAR collapses at the profile angle: not closed
    assert not s.eyes_closed
    st, t = calibrated()
    s, t = run(st, 2.0, t, pitch=-40, ear=0.05, blink=0.1)
    assert not s.eyes_closed


def test_driver_not_visible_warns_then_alerts():
    st, t = calibrated()
    s, t = run(st, 2.5, t, face=False)
    assert s.level == Level.WARN and "Driver not visible" in s.reasons
    s, t = run(st, 3.0, t, face=False)
    assert s.level == Level.ALERT and "Driver not visible" in s.reasons

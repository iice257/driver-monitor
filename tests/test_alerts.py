from driver_monitor.alarm import Alarm
from driver_monitor.alerts import VoicePolicy, cue_for

FPS = 30


def speak(policy, seconds, reasons, start=0.0, yawns=None):
    """Feed the same reasons for a while; returns [(t, phrase)] that were spoken."""
    said = []
    for i in range(int(seconds * FPS)):
        t = start + i / FPS
        cue = policy.update(reasons, t, yawns)
        if cue:
            said.append((round(t, 2), cue.phrase))
    return said, start + seconds


def test_every_reason_names_the_problem_then_asks_politely():
    for reason, phrase in [
        ("Eyes closed 1.3s", "Eyes closed, please stay awake."),
        ("Phone in use", "Phone detected, please put your phone away."),
        ("Phone call", "Phone call detected, please keep your eyes on the road."),
        ("Eyes off road", "Driver distracted, please focus on the road."),
        ("Driver not visible", "Driver not visible, please pay attention."),
        ("Fatigue: PERCLOS 18%", "Fatigue detected, please take a break soon."),
        ("Fatigue: 3 yawns", "Fatigue detected, please take a break soon."),
        ("Yawn", "Yawning detected, please drive carefully."),
    ]:
        assert cue_for(reason).phrase == phrase
    assert cue_for("Something new") is None


def test_spoken_once_on_onset_then_at_the_repeat_interval():
    p = VoicePolicy()
    said, _ = speak(p, 9.0, ["Eyes closed 2.0s"])
    assert [t for t, _ in said] == [0.0, 4.0, 8.0]


def test_numbers_in_the_reason_do_not_retrigger():
    p = VoicePolicy()
    said = [p.update([f"Eyes closed {1.2 + i / FPS:.1f}s"], i / FPS) for i in range(60)]
    assert sum(c is not None for c in said) == 1


def test_flicker_is_rate_limited():
    p = VoicePolicy()
    assert p.update(["Eyes off road"], 0.0)
    assert p.update([], 0.5) is None
    assert p.update(["Eyes off road"], 1.0) is None  # back within min_gap_s
    assert p.update(["Eyes off road"], 3.0)  # pending onset is spoken once the gap has passed


def test_one_phrase_at_a_time_in_priority_order():
    p = VoicePolicy()
    first = p.update(["Eyes closed 1.3s", "Eyes off road"], 0.0)
    second = p.update(["Eyes closed 1.4s", "Eyes off road"], 0.1)
    assert (first.key, second.key) == ("eyes", "away")


def test_each_new_yawn_is_announced():
    p = VoicePolicy()
    said, t = speak(p, 3.0, ["Yawn"], yawns=1)
    said2, t = speak(p, 3.0, ["Yawn"], start=t, yawns=2)
    said3, _ = speak(p, 3.0, ["Fatigue: 3 yawns"], start=t, yawns=3)
    assert [s.split(",")[0] for _, s in said + said2 + said3] == ["Yawning detected", "Yawning detected", "Fatigue detected"]


def test_reset_forgets_history():
    p = VoicePolicy()
    assert p.update(["Phone in use"], 0.0)
    p.reset()
    assert p.update(["Phone in use"], 0.1)


def test_disabled_alarm_still_reports_the_cue_without_sound():
    a = Alarm(enabled=False)
    cue = a.update(["Yawn"], 0.0, urgent=False, yawns=1)
    assert cue.key == "yawn"
    a.close()


def test_every_cue_has_a_current_voice_clip_and_stale_ones_are_ignored(tmp_path):
    from driver_monitor.alerts import CUES, Cue
    from driver_monitor.voice import clip_for

    for _, cue in CUES:
        assert clip_for(cue) is not None, f"re-render voice clips: python -m driver_monitor.voice ({cue.key})"
    (tmp_path / "yawn.wav").write_bytes(b"")
    (tmp_path / "manifest.json").write_text('{"yawn": "Old wording"}', encoding="utf-8")
    assert clip_for(Cue("yawn", "Y", "New wording", 1.0), tmp_path) is None


def run(policy, seconds, reasons, urgent, start=0.0):
    out = []
    for i in range(int(seconds * FPS)):
        t = start + i / FPS
        d = policy.update(reasons, t, urgent)
        if d.cue or d.siren_hz:
            out.append((round(t, 2), d))
    return out


def test_alert_cues_preempt_and_warn_cues_do_not():
    from driver_monitor.alerts import AlarmPolicy

    p = AlarmPolicy()
    first = run(p, 0.1, ["Eyes off road"], urgent=True)[0][1]
    assert first.cue.key == "away" and first.preempt
    p = AlarmPolicy()
    first = run(p, 0.1, ["Yawn"], urgent=False)[0][1]
    assert first.cue.key == "yawn" and not first.preempt


def test_closed_eyes_escalate_to_a_faster_siren_after_two_seconds():
    from driver_monitor.alerts import AlarmPolicy

    p = AlarmPolicy()
    events = run(p, 9.0, ["Eyes closed 1.0s"], urgent=True)
    phrases = [(t, d) for t, d in events if d.cue]
    sirens = [(t, d.siren_hz) for t, d in events if d.siren_hz]
    assert [t for t, _ in phrases] == [0.0] and phrases[0][1].preempt  # one phrase, then the siren
    assert 1.9 <= sirens[0][0] <= 2.1
    assert sirens[0][1] == 2.0 and sirens[-1][1] > sirens[0][1]
    assert sirens[-1][1] <= p.siren_max_hz
    rates = [hz for _, hz in sirens]
    assert rates == sorted(rates)  # never slows down while the eyes stay shut


def test_siren_stops_when_eyes_open_and_restarts_from_the_bottom():
    from driver_monitor.alerts import AlarmPolicy

    p = AlarmPolicy()
    run(p, 5.0, ["Eyes closed 1.0s"], urgent=True)
    assert p.update([], 5.1, False).siren_hz == 0
    again = run(p, 3.0, ["Eyes closed 1.0s"], urgent=True, start=6.0)
    assert [d.siren_hz for _, d in again if d.siren_hz][0] == 2.0


def test_phone_watcher_reports_a_failing_backend_instead_of_dying(monkeypatch):
    import time

    import numpy as np

    from driver_monitor import phone

    class Broken:
        def __init__(self, score):
            pass

        def __call__(self, frame):
            raise RuntimeError("delegate lost")

        def close(self):
            pass

    monkeypatch.setattr(phone, "_MediaPipeBackend", Broken)
    w = phone.PhoneWatcher()
    w.submit(np.zeros((8, 8, 3), np.uint8), 0.0)
    for _ in range(50):
        if w.error:
            break
        time.sleep(0.05)
    assert "delegate lost" in w.error
    assert w.latest().phones == []
    w.close()


def test_phone_watcher_survives_a_backend_that_cannot_be_built(monkeypatch):
    from driver_monitor import phone

    def boom(score):
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(phone, "_MediaPipeBackend", boom)
    w = phone.PhoneWatcher()
    assert "model missing" in w.error
    w.close()


def test_recording_is_h264_with_the_spoken_alerts_mixed_in(tmp_path):
    import shutil
    import subprocess

    import pytest

    from driver_monitor.reel import encode_recording, probe

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    raw = tmp_path / "raw.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=6",
                    "-pix_fmt", "yuv420p", str(raw)], check=True)
    out = tmp_path / "out.mp4"
    placed = encode_recording(raw, None, [(1.0, "Yawning detected, please drive carefully.")], out)
    assert placed == [(1.0, "Yawning detected, please drive carefully.")]
    codecs = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(out)],
                            capture_output=True, text=True, check=True).stdout.split()
    assert set(codecs) == {"h264", "aac"}
    assert probe(out)[1]

"""Command-line entry point: python -m driver_monitor [--source 0|video.mp4]"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

from . import hud
from .alarm import Alarm
from .detector import DEFAULT_MODEL, FaceTracker
from .phone import NullWatcher, PhoneWatcher, hand_at_ear, phone_near_face
from .state import Config, DriverState, Frame, Level

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.toml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="driver_monitor", description="Real-time driver drowsiness and distraction monitor.")
    p.add_argument("--source", default="0", help="camera index or path to a video file (default: 0)")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="thresholds file (default: config.toml)")
    p.add_argument("--phone", choices=["mediapipe", "yolo", "off"], default="mediapipe",
                   help="phone detection backend (yolo needs: pip install ultralytics)")
    p.add_argument("--no-alarm", action="store_true", help="disable the spoken / audible alarm")
    p.add_argument("--beep", action="store_true", help="beep instead of speaking the alert")
    p.add_argument("--headless", action="store_true", help="no window; useful with --log on a video file")
    p.add_argument("--record", type=Path, help="write the annotated video (H.264, with the spoken alerts mixed in) to this .mp4; needs ffmpeg")
    p.add_argument("--log", type=Path, help="append state changes as JSON lines to this file")
    p.add_argument("--mirror", action="store_true", help="flip webcam horizontally for display")
    p.add_argument("--recalibrate-at", default="", metavar="T1,T2",
                   help="seconds at which to recalibrate, e.g. the cuts in a spliced test reel")
    return p.parse_args()


def _finish_recording(raw: Path, out: Path, source: str | None, said: list[tuple[float, str]]) -> None:
    """OpenCV can only write mp4v: re-encode to H.264 and mix in the spoken alerts (and the input's sound)."""
    from .reel import encode_recording

    try:
        placed = encode_recording(raw, Path(source) if source else None, said, out)
        raw.unlink(missing_ok=True)
        print(f"Recorded {out} (H.264, {len(placed)} spoken alerts)")
    except (OSError, subprocess.CalledProcessError) as exc:  # no ffmpeg: keep the raw recording
        fallback = out.with_name(out.stem + ".mp4v.mp4")
        raw.replace(fallback)
        print(f"ffmpeg unavailable ({exc}); kept silent mp4v recording at {fallback}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    is_file = not isinstance(source, int)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {args.source!r}")

    cfg = Config.load(args.config)
    state = DriverState(cfg)
    alarm = Alarm(enabled=not args.no_alarm and not args.headless, voice=not args.beep)
    watcher = NullWatcher() if args.phone == "off" else PhoneWatcher(args.phone)
    writer = None
    log = args.log.open("a", encoding="utf-8") if args.log else None
    last_key: tuple | None = None
    phone_reported = False
    said: list[tuple[float, str]] = []   # spoken alerts, mixed into --record
    raw = args.record.with_suffix(".raw.mp4") if args.record else None
    t0 = time.monotonic()
    fps, prev = 0.0, time.monotonic()
    recal = sorted(float(x) for x in args.recalibrate_at.split(",") if x.strip())

    try:
        with FaceTracker(args.model) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.mirror:
                    frame = cv2.flip(frame, 1)

                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_file else time.monotonic() - t0
                if recal and t >= recal[0]:
                    recal.pop(0)
                    state.recalibrate()
                    alarm.reset()
                face = tracker.process(frame, int(t * 1000))
                if watcher.error and not phone_reported:
                    phone_reported = True
                    print(f"Phone detection off: {watcher.error}", file=sys.stderr)
                    if log:
                        log.write(json.dumps({"t": round(t, 2), "level": "INFO", "reasons": ["Phone detection off"]}) + "\n")
                watcher.submit(frame, t)
                phone = watcher.latest()
                fresh = abs(t - phone.t) < 1.0

                f = Frame(t=t, face=face is not None, phone=fresh and bool(phone.phones))
                if face is not None:
                    f.ear, f.mar, f.pitch, f.yaw, f.gaze = face.ear, face.mar, face.pitch, face.yaw, face.gaze
                    if face.blendshapes:
                        f.blink, f.jaw_open = face.blink, face.jaw_open
                    if fresh:
                        f.phone_near_face = phone_near_face(phone.phones, face.box)
                        f.hand_at_ear = any(hand_at_ear(hd, face.box) for hd in phone.hands)
                status = state.update(f)
                cue = alarm.update(status.reasons, t, urgent=status.level == Level.ALERT, yawns=status.yawns)
                if cue is not None:
                    said.append((t, cue.phrase))

                key = (status.level, tuple(r.split(" ")[0] for r in status.reasons))
                if log and key != last_key:
                    log.write(json.dumps({"t": round(t, 2), "level": status.level.name, "reasons": status.reasons}) + "\n")
                    last_key = key

                now = time.monotonic()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
                prev = now
                hud.draw(frame, face, status, fps, phone if fresh else None, cfg, t, phone_off=bool(watcher.error))

                if args.record:
                    if writer is None:
                        h, w = frame.shape[:2]
                        rate = cap.get(cv2.CAP_PROP_FPS) or 30
                        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), rate, (w, h))
                    writer.write(frame)

                if not args.headless:
                    cv2.imshow("Driver Monitor  [q] quit  [c] recalibrate", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        break
                    if k == ord("c"):
                        state.recalibrate()
                        alarm.reset()
    finally:
        watcher.close()
        alarm.close()
        cap.release()
        if writer:
            writer.release()
            _finish_recording(raw, args.record, source if is_file else None, said)
        if log:
            log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

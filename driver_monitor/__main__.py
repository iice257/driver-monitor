"""Command-line entry point: python -m driver_monitor [--source 0|video.mp4]"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from . import hud
from .alarm import Alarm
from .detector import DEFAULT_MODEL, FaceTracker
from .state import Config, DriverState, Frame, Level


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="driver_monitor", description="Real-time driver drowsiness and distraction monitor.")
    p.add_argument("--source", default="0", help="camera index or path to a video file (default: 0)")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--no-alarm", action="store_true", help="disable the audible alarm")
    p.add_argument("--headless", action="store_true", help="no window; useful with --log on a video file")
    p.add_argument("--record", type=Path, help="write the annotated video to this .mp4")
    p.add_argument("--log", type=Path, help="append state changes as JSON lines to this file")
    p.add_argument("--mirror", action="store_true", help="flip webcam horizontally for display")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    is_file = not isinstance(source, int)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {args.source!r}")

    state = DriverState(Config())
    alarm = Alarm(enabled=not args.no_alarm and not args.headless)
    writer = None
    log = args.log.open("a", encoding="utf-8") if args.log else None
    last_key: tuple | None = None
    t0 = time.monotonic()
    fps, prev = 0.0, time.monotonic()

    with FaceTracker(args.model) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_file else time.monotonic() - t0
            face = tracker.process(frame, int(t * 1000))
            status = state.update(
                Frame(t=t, face=face is not None, ear=face.ear if face else 0.0,
                      mar=face.mar if face else 0.0, pitch=face.pitch if face else 0.0,
                      yaw=face.yaw if face else 0.0)
            )
            alarm.set(status.level == Level.ALERT)

            key = (status.level, tuple(r.split(" ")[0] for r in status.reasons))
            if log and key != last_key and not status.calibrating:
                log.write(json.dumps({"t": round(t, 2), "level": status.level.name, "reasons": status.reasons}) + "\n")
                last_key = key

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            hud.draw(frame, face, status, fps)

            if args.record:
                if writer is None:
                    h, w = frame.shape[:2]
                    rate = cap.get(cv2.CAP_PROP_FPS) or 30
                    writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), rate, (w, h))
                writer.write(frame)

            if not args.headless:
                cv2.imshow("Driver Monitor  [q] quit  [c] recalibrate", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("c"):
                    state.recalibrate()

    alarm.close()
    cap.release()
    if writer:
        writer.release()
    if log:
        log.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Scenario check: run labelled clips through the monitor and compare against what should happen.

    python -m driver_monitor.evaluate samples/scenarios.json

Each scenario names a clip, the alerts that must appear (optionally inside a time
window) and the ones that must not. The clips in samples/ are AI-generated
driver-camera footage, so nobody real is on camera.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2

from .detector import DEFAULT_MODEL, FaceTracker
from .state import Config, DriverState, Frame, Level

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.toml"


def frames(path: Path, model: Path = DEFAULT_MODEL) -> list[Frame]:
    """The per-frame signal values (what DriverState consumes) for a video file, phone detection off."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    out = []
    with FaceTracker(model) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            face = tracker.process(frame, int(t * 1000))
            f = Frame(t=t, face=face is not None)
            if face is not None:
                f.ear, f.mar, f.pitch, f.yaw, f.gaze = face.ear, face.mar, face.pitch, face.yaw, face.gaze
                if face.blendshapes:
                    f.blink, f.jaw_open = face.blink, face.jaw_open
            out.append(f)
    cap.release()
    return out


def run_state(cfg: Config, signal: list[Frame]) -> list[tuple[float, Level, list[str]]]:
    state = DriverState(cfg)
    out = []
    for f in signal:
        s = state.update(f)
        out.append((f.t, s.level, s.reasons))
    return out


def timeline(path: Path, cfg: Config, model: Path = DEFAULT_MODEL) -> list[tuple[float, Level, list[str]]]:
    """Every frame's (t, level, reasons) for a video file, phone detection off."""
    return run_state(cfg, frames(path, model))


def _seen(tl, prefix: str, window=None) -> list[float]:
    lo, hi = window or (0.0, float("inf"))
    return [t for t, _, reasons in tl if lo <= t <= hi and any(r.startswith(prefix) for r in reasons)]


def _onsets(tl, reasons: list[str]) -> int:
    """How many separate times any of the reasons switched on."""
    count, prev = 0, False
    for _, _, rs in tl:
        now = any(r.startswith(x) for r in rs for x in reasons)
        count += now and not prev
        prev = now
    return count


def check(scenario: dict, tl) -> list[str]:
    problems = []
    for exp in scenario.get("expect", []):
        hits = _seen(tl, exp["reason"], exp.get("within"))
        if not hits:
            where = f" between {exp['within'][0]}s and {exp['within'][1]}s" if "within" in exp else ""
            problems.append(f"missing '{exp['reason']}'{where}")
    for reason in scenario.get("forbid", []):
        hits = _seen(tl, reason)
        if hits:
            problems.append(f"false '{reason}' at {hits[0]:.1f}s")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(prog="driver_monitor.evaluate", description=__doc__.splitlines()[0])
    p.add_argument("spec", type=Path, help="scenarios JSON")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()

    cfg = Config.load(args.config)
    scenarios = json.loads(args.spec.read_text(encoding="utf-8"))
    failed = 0
    neg_seconds, neg_alarms = 0.0, 0
    for sc in scenarios:
        tl = timeline(args.spec.parent / sc["clip"], cfg)
        problems = check(sc, tl)
        failed += bool(problems)
        events = sorted({re.sub(r" [\d.]+s?%?$", "", r) for _, _, rs in tl for r in rs})
        print(f"{'PASS' if not problems else 'FAIL'}  {sc['label']:<22} {sc['clip']:<16} saw: {', '.join(events) or 'nothing'}")
        for exp in sc.get("expect", []):
            hits = _seen(tl, exp["reason"], exp.get("within"))
            print(f"        expect {exp['reason']!r}: {len(hits)} frames" + (f", first at {hits[0]:.1f}s" if hits else ""))
        for reason in sc.get("forbid", []):
            hits = _seen(tl, reason)
            print(f"        forbid {reason!r}: {len(hits)} frames")
        if sc.get("negative"):  # normal driving: every forbidden onset is a false alarm
            neg_seconds += tl[-1][0] if tl else 0.0
            neg_alarms += _onsets(tl, sc.get("forbid", []))
        for msg in problems:
            print(f"        {msg}")
    print(f"\n{len(scenarios) - failed}/{len(scenarios)} scenarios passed")
    if neg_seconds:
        print(f"False alarms on normal driving: {neg_alarms} in {neg_seconds:.0f} s = {neg_alarms / neg_seconds * 60:.2f} per minute")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

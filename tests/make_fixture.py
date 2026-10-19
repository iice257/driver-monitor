"""Regenerate tests/fixtures/signal_trace.json from the sample clips.

    python -m tests.make_fixture

The trace is the per-frame signal (EAR, jaw, pose, ...) that DriverState consumes, so the Python
and JavaScript state machines can be compared without any model. Real clips supply the face
signals; phone, hand-at-ear and a missing face are injected into copies so those paths run too.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from driver_monitor.evaluate import frames

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("t", "face", "ear", "mar", "pitch", "yaw", "gaze", "blink", "jaw_open", "phone", "phone_near_face", "hand_at_ear")


def dump(signal):
    return [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in dataclasses.asdict(f).items() if k in KEYS} for f in signal]


def inject(signal, **windows):
    out = [dataclasses.replace(f) for f in signal]
    for f in out:
        for name, (lo, hi) in windows.items():
            if lo <= f.t < hi:
                if name == "no_face":
                    f.face = False
                else:
                    setattr(f, name, True)
    return out


def main() -> None:
    traces = {name: frames(ROOT / "samples" / f"{name}.mp4") for name in ("drowsy", "yawn", "lookaway")}
    base = traces["lookaway"]
    span = base[-1].t + 1 / 30
    looped = [dataclasses.replace(f, t=f.t + k * span) for k in range(3) for f in base]
    traces["injected"] = inject(looped, phone=(2.0, 6.0), phone_near_face=(2.5, 6.0),
                                hand_at_ear=(9.0, 12.0), no_face=(16.0, 23.0))
    out = ROOT / "tests" / "fixtures" / "signal_trace.json"
    out.write_text(json.dumps({k: dump(v) for k, v in traces.items()}, separators=(",", ":")), encoding="utf-8")
    print(out, {k: len(v) for k, v in traces.items()})


if __name__ == "__main__":
    main()

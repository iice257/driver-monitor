"""The browser state machine (web/engine.js) must decide exactly like driver_monitor/state.py."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from driver_monitor.state import Config, DriverState, Frame

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "tests" / "fixtures" / "signal_trace.json"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_python_and_javascript_agree_on_every_frame():
    traces = json.loads(TRACE.read_text(encoding="utf-8"))
    run = subprocess.run(["node", str(ROOT / "tests" / "parity_node.mjs")], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    js = json.loads(run.stdout)
    seen_alert = False
    for name, rows in traces.items():
        state = DriverState(Config())
        assert len(js[name]) == len(rows)
        for i, row in enumerate(rows):
            s = state.update(Frame(**row))
            seen_alert |= s.level >= 2
            assert [int(s.level), s.reasons] == js[name][i], f"{name} frame {i} t={row['t']}"
    assert seen_alert, "the trace should exercise at least one alert"


def test_trace_covers_phone_and_missing_face_paths():
    injected = json.loads(TRACE.read_text(encoding="utf-8"))["injected"]
    assert any(f["phone"] for f in injected) and any(f["hand_at_ear"] for f in injected)
    assert any(not f["face"] for f in injected)

import numpy as np
import pytest

from driver_monitor import hud
from driver_monitor.state import Config, DriverState, Frame, Level


def status_after(seconds, **signal):
    state, s = DriverState(Config()), None
    for i in range(int(seconds * 30)):
        s = state.update(Frame(t=i / 30, face=True, ear=0.3, **signal))
    return s


@pytest.mark.parametrize("size", [(1280, 720), (854, 480), (1920, 1080)])
def test_panel_stays_inside_the_frame_at_every_size(size):
    w, h = size
    img = np.full((h, w, 3), 128, np.uint8)
    hud.draw(img, None, status_after(5), 30.0, cfg=Config())
    changed = np.argwhere((img != 128).any(axis=2))
    assert changed.size, "nothing was drawn"
    panel = img[: h // 2, w - w // 4 :]
    assert (panel != 128).any(), "the telemetry panel is missing from the top right"
    assert changed[:, 1].max() <= w - 1 and changed[:, 0].max() <= h - 1


def test_banner_detail_adds_information_or_is_absent():
    s = status_after(2.0, blink=0.0)
    assert hud._detail(s, "X") is None  # attentive: nothing to say
    yawn = status_after(0.5)
    yawn.reasons = ["Yawn"]
    assert hud._detail(yawn, "YAWNING DETECTED") is None  # "Yawn" would just repeat the headline
    eyes = status_after(0.5)
    eyes.reasons = ["Eyes closed 1.4s"]
    assert hud._detail(eyes, "EYES CLOSED") == "Closed for 1.4 s"
    assert eyes.level in (Level.OK, Level.WARN, Level.ALERT)


def test_labels_match_the_web_inspector():
    import re
    from pathlib import Path

    js = (Path(__file__).resolve().parent.parent / "web" / "app.js").read_text(encoding="utf-8")
    web = re.findall(r'label: "([^"]+)", lo:', js)
    src = Path(hud.__file__).read_text(encoding="utf-8")
    for label in web:
        assert f'"{label}"' in src, label

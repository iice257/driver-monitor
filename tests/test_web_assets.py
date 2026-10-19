"""The web demo is self-contained: pinned runtime and models under web/, sizes the progress bar relies on."""

import re
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
WORKER = (WEB / "worker.js").read_text(encoding="utf-8")


def asset_paths():
    vendor = re.search(r'const VENDOR = `([^`]*)`', WORKER).group(1).replace("${VERSION}", re.search(r'VERSION = "([^"]+)"', WORKER).group(1))
    for url, size in re.findall(r"url: `([^`]+)`, bytes: (\d+)", WORKER):
        yield WEB / url.replace("${VENDOR}", vendor).replace("${MODEL_DIR}", "models").lstrip("/").replace("//", "/"), int(size), url


def test_every_listed_asset_exists_with_the_declared_size():
    found = list(asset_paths())
    assert len(found) >= 6
    for path, size, url in found:
        p = Path(str(path).replace("\\", "/").replace("/web//", "/web/"))
        assert p.exists(), url
        assert p.stat().st_size == size, f"{url}: {p.stat().st_size} on disk, {size} in worker.js"


def test_nothing_is_fetched_from_a_cdn_at_runtime():
    for name in ("worker.js", "app.js", "engine.js"):
        text = (WEB / name).read_text(encoding="utf-8")
        assert "jsdelivr" not in text and "googleapis" not in text, name


def test_same_model_files_as_the_desktop_app():
    from driver_monitor import detector, phone

    assert phone.OBJECT_MODEL.name == "efficientdet_lite0.tflite" and "efficientdet_lite0/float16" in phone.OBJECT_MODEL_URL
    assert phone.HAND_MODEL.name == "hand_landmarker.task"
    assert detector.DEFAULT_MODEL.name == "face_landmarker.task"
    for name in (phone.OBJECT_MODEL.name, phone.HAND_MODEL.name, detector.DEFAULT_MODEL.name):
        assert (WEB / "models" / name).exists(), name


def test_browser_scripts_parse_as_modules(tmp_path):
    """app.js and engine.js are ES modules: a duplicate `const` only fails in the browser, so parse them here."""
    import shutil
    import subprocess

    import pytest

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    for name in ("app", "engine"):
        copy = tmp_path / f"{name}.mjs"
        copy.write_text((WEB / f"{name}.js").read_text(encoding="utf-8"), encoding="utf-8")
        run = subprocess.run(["node", "--check", str(copy)], capture_output=True, text=True)
        assert run.returncode == 0, f"{name}.js: {run.stderr}"

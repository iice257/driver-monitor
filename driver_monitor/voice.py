"""Pre-render every alert phrase in a natural neural voice.

    pip install edge-tts
    python -m driver_monitor.voice [--voice en-US-AvaNeural]

Rendering needs a network connection once (Microsoft Edge's neural voices). The clips
are committed, so the monitor and the web demo then play them offline:
  driver_monitor/voice/<cue>.wav   played by the desktop alarm and mixed into the reel
  web/voice/<cue>.mp3              played by the web demo
manifest.json records the phrase each clip says, so a phrase edited in alerts.py is
never played from a stale clip: it falls back to system TTS until re-rendered.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

from .alerts import CUES, Cue

VOICE_DIR = Path(__file__).resolve().parent / "voice"
WEB_VOICE_DIR = Path(__file__).resolve().parent.parent / "web" / "voice"
DEFAULT_VOICE = "en-US-AvaNeural"


def _manifest(folder: Path) -> dict[str, str]:
    try:
        return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def clip_for(cue: Cue, folder: Path = VOICE_DIR, ext: str = ".wav") -> Path | None:
    """The rendered clip for this cue, if it exists and still says the current phrase."""
    path = folder / f"{cue.key}{ext}"
    if path.exists() and _manifest(folder).get(cue.key) == cue.phrase:
        return path
    return None


async def _render(voice: str, rate: str) -> None:
    import edge_tts

    cues = {cue.key: cue for _, cue in CUES}
    VOICE_DIR.mkdir(exist_ok=True)
    WEB_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for key, cue in cues.items():
            raw = Path(tmp) / f"{key}.mp3"
            await edge_tts.Communicate(cue.phrase, voice, rate=rate).save(str(raw))
            # Trim edge silence, then a 48 kHz WAV for playback/mixing and a small MP3 for the web.
            trim = "silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse"
            for out, codec in ((VOICE_DIR / f"{key}.wav", ["-c:a", "pcm_s16le", "-ar", "48000"]),
                               (WEB_VOICE_DIR / f"{key}.mp3", ["-c:a", "libmp3lame", "-b:a", "64k", "-ar", "24000"])):
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-af", trim, "-ac", "1", *codec, str(out)],
                               check=True)
            print(f"{key:8} {cue.phrase}")
    manifest = json.dumps({k: c.phrase for k, c in cues.items()}, indent=2) + "\n"
    for folder in (VOICE_DIR, WEB_VOICE_DIR):
        (folder / "manifest.json").write_text(manifest, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(prog="driver_monitor.voice", description=__doc__.splitlines()[0])
    p.add_argument("--voice", default=DEFAULT_VOICE, help="edge-tts voice name (edge-tts --list-voices)")
    p.add_argument("--rate", default="+0%", help="speaking rate, e.g. -5%%")
    args = p.parse_args()
    asyncio.run(_render(args.voice, args.rate))


if __name__ == "__main__":
    main()

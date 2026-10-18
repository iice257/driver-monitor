"""Build the demo reel from the scenario clips.

    python -m driver_monitor.reel samples/scenarios.json

Writes, next to the scenarios file:
  reel.mp4      the clips as one continuous drive: dissolves at the cuts and a single
                uniform cabin ambience underneath. The web demo analyses this one live,
                so it carries no overlay.
  showcase.mp4  the same reel run through the monitor: HUD, alert banner and flashing
                border drawn on every frame, and each alert spoken in the pre-rendered
                voice (voice.py) over the ambience, exactly when the live alarm would.

With --web DIR it also refreshes DIR/reel.mp4, DIR/showcase.mp4, DIR/poster.jpg and
DIR/chapters.json for the static site. Phone detection is off, as in evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2

from . import hud
from .alerts import CUES, VoicePolicy
from .detector import DEFAULT_MODEL, FaceTracker
from .evaluate import DEFAULT_CONFIG
from .state import Config, DriverState, Frame
from .voice import clip_for

W, H, FPS, RATE = 1280, 720, 30, 48000


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def probe(path: Path) -> tuple[float, bool]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return float(info["format"]["duration"]), has_audio


def join(clips: list[Path], out: Path, ambience: Path, xfade: float = 0.5) -> list[float]:
    """Join the clips into one continuous take; returns where each scenario starts.

    Picture: a short dissolve at every cut. Every clip opens on the same reference frame,
    so the dissolve reads as the same drive rather than a new shot.
    Sound: one uniform cabin ambience under the whole reel instead of each clip's own
    track, which differ in level and timbre and give the cuts away. The ambience clip's
    sound is looped with long crossfades that fall between the cuts, then normalised.
    """
    durs = [probe(c)[0] for c in clips]
    total = sum(durs) - xfade * (len(clips) - 1)
    starts = [sum(durs[:i]) - xfade * i + (xfade / 2 if i else 0.0) for i in range(len(clips))]

    inputs, chains = [], []
    for i, clip in enumerate(clips):
        inputs += ["-i", str(clip)]
        chains.append(f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:-1:-1,"
                      f"fps={FPS},setsar=1,format=yuv420p,trim=duration={durs[i]},setpts=PTS-STARTPTS[v{i}]")
    last, offset = "[v0]", 0.0
    for i in range(1, len(clips)):
        offset += durs[i - 1] - xfade
        chains.append(f"{last}[v{i}]xfade=transition=fade:duration={xfade}:offset={offset:.3f}[x{i}]")
        last = f"[x{i}]"

    # Ambience bed: overlapping copies of one clip's sound, 3 s equal-power crossfades.
    amb_dur, has_audio = probe(ambience)
    if not has_audio:
        raise SystemExit(f"{ambience} has no sound to use as ambience")
    seg, overlap = amb_dur - 0.4, 3.0
    copies = 1
    while copies * seg - (copies - 1) * overlap < total + 1:
        copies += 1
    a0 = len(clips)
    for k in range(copies):
        inputs += ["-i", str(ambience)]
        chains.append(f"[{a0 + k}:a]aformat=sample_rates={RATE}:channel_layouts=stereo,"
                      f"atrim=start=0.2:duration={seg},asetpts=PTS-STARTPTS[b{k}]")
    bed = "[b0]"
    for k in range(1, copies):
        chains.append(f"{bed}[b{k}]acrossfade=d={overlap}:c1=qsin:c2=qsin[bx{k}]")
        bed = f"[bx{k}]"
    chains.append(f"{bed}atrim=duration={total},asetpts=PTS-STARTPTS,loudnorm=I=-30:LRA=4:TP=-6,"
                  f"aresample={RATE},afade=t=in:d=0.6,afade=t=out:st={total - 1.0:.3f}:d=1.0[a]")

    ffmpeg(*inputs, "-filter_complex", ";".join(chains), "-map", last, "-map", "[a]",
           "-c:v", "libx264", "-crf", "22", "-preset", "slow", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out))
    return starts


def annotate(reel: Path, out: Path, cuts: list[float], cfg: Config, model: Path) -> list[tuple[float, str]]:
    """Run the monitor over the reel, drawing the HUD. Returns (t, phrase) for every spoken alert."""
    cap = cv2.VideoCapture(str(reel))
    rate = cap.get(cv2.CAP_PROP_FPS) or FPS
    writer = None
    state, voice, said = DriverState(cfg), VoicePolicy(), []
    pending = sorted(c for c in cuts if c > 0)
    with FaceTracker(model) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pending and t >= pending[0] - 1e-3:
                pending.pop(0)
                state.recalibrate()  # a new scenario is a new driver, as in the web demo
                voice.reset()
            face = tracker.process(frame, int(t * 1000))
            f = Frame(t=t, face=face is not None)
            if face is not None:
                f.ear, f.mar, f.pitch, f.yaw, f.gaze = face.ear, face.mar, face.pitch, face.yaw, face.gaze
                if face.blendshapes:
                    f.blink, f.jaw_open = face.blink, face.jaw_open
            s = state.update(f)
            if not s.calibrating:
                cue = voice.update(s.reasons, t, s.yawns)
                if cue is not None:
                    said.append((t, cue.phrase))
            hud.draw(frame, face, s, rate, None, cfg, t)
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), rate, frame.shape[1::-1])
            writer.write(frame)
    cap.release()
    if writer:
        writer.release()
    return said


def synthesize(phrases: set[str], folder: Path, rate: int = 185) -> dict[str, Path]:
    """A WAV per phrase: the pre-rendered neural voice clip, or system TTS if it is stale."""
    by_phrase = {cue.phrase: cue for _, cue in CUES}
    files, missing = {}, []
    for phrase in sorted(phrases):
        clip = clip_for(by_phrase[phrase]) if phrase in by_phrase else None
        if clip is not None:
            files[phrase] = clip
        else:
            missing.append(phrase)
    if missing:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        for i, phrase in enumerate(missing):
            files[phrase] = folder / f"say{i}.wav"
            engine.save_to_file(phrase, str(files[phrase]))
        engine.runAndWait()
    return files


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def mix(video: Path, reel: Path, said: list[tuple[float, str]], voices: dict[str, Path], out: Path) -> list[tuple[float, str]]:
    """Mux the annotated video with the reel's sound, ducked, plus each phrase at its time."""
    inputs = ["-i", str(video), "-i", str(reel)]
    chains = [f"[1:a]volume=0.7[bed]"]
    labels, placed, free_at = ["[bed]"], [], 0.0
    for i, (t, phrase) in enumerate(said):
        at = max(t, free_at)  # one voice: never talk over the previous phrase
        free_at = at + wav_seconds(voices[phrase]) + 0.2
        placed.append((round(at, 2), phrase))
        inputs += ["-i", str(voices[phrase])]
        ms = int(at * 1000)
        chains.append(f"[{i + 2}:a]aformat=sample_rates={RATE}:channel_layouts=stereo,"
                      f"adelay={ms}|{ms},volume=1.8[s{i}]")
        labels.append(f"[s{i}]")
    chains.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0,alimiter=limit=0.95[a]")
    ffmpeg(*inputs, "-filter_complex", ";".join(chains), "-map", "0:v", "-map", "[a]",
           "-c:v", "libx264", "-crf", "22", "-preset", "slow", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out))
    return placed


def main() -> None:
    p = argparse.ArgumentParser(prog="driver_monitor.reel", description=__doc__.splitlines()[0])
    p.add_argument("spec", type=Path, help="scenarios JSON (clip order = reel order)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--web", type=Path, help="also refresh the static site's demo folder, e.g. web/demo")
    p.add_argument("--ambience", default=None, help="clip whose sound becomes the reel's ambience (default: the first)")
    p.add_argument("--xfade", type=float, default=0.5, help="dissolve between scenarios, seconds")
    args = p.parse_args()

    folder = args.spec.parent
    scenarios = json.loads(args.spec.read_text(encoding="utf-8"))
    clips = [folder / sc["clip"] for sc in scenarios]
    reel, showcase = folder / "reel.mp4", folder / "showcase.mp4"

    ambience = folder / args.ambience if args.ambience else clips[0]
    starts = join(clips, reel, ambience, args.xfade)
    total, _ = probe(reel)
    print(f"reel.mp4      {total:.1f}s, {len(clips)} scenarios, one ambience track ({ambience.name})")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        said = annotate(reel, tmp / "hud.mp4", starts, Config.load(args.config), args.model)
        voices = synthesize({phrase for _, phrase in said}, tmp)
        placed = mix(tmp / "hud.mp4", reel, said, voices, showcase)
    print(f"showcase.mp4  {len(placed)} spoken alerts")
    for t, phrase in placed:
        print(f"  {t:5.1f}s  {phrase}")

    if args.web:
        args.web.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reel, args.web / "reel.mp4")
        shutil.copyfile(showcase, args.web / "showcase.mp4")
        ffmpeg("-ss", "1.5", "-i", str(reel), "-frames:v", "1", "-q:v", "3", str(args.web / "poster.jpg"))
        ends = starts[1:] + [total]
        chapters = [{"label": sc["label"], "start": round(s, 2), "end": round(e, 2), "expect": sc.get("summary", "")}
                    for sc, s, e in zip(scenarios, starts, ends)]
        (args.web / "chapters.json").write_text(
            "[\n" + ",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in chapters) + "\n]\n", encoding="utf-8")
        print(f"refreshed {args.web}: reel.mp4, showcase.mp4, poster.jpg, chapters.json")


if __name__ == "__main__":
    main()

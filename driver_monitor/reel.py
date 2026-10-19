"""Build the demo reel from the scenario clips.

    python -m driver_monitor.reel samples/scenarios.json

Writes to web/demo (or --out):
  reel.mp4      the clips as one continuous, seamlessly looping drive: muted, mirrored so
                the driver sits on the left, cut where the poses match, dissolved, with
                one steady cabin ambience underneath. The web demo analyses this one
                live, so it carries no overlay. A scenario's "min_end" keeps its clip
                running at least that long.
  showcase.mp4  the same reel run through the monitor: HUD, alert banner and flashing
                border drawn on every frame, and each alert spoken in the pre-rendered
                voice (voice.py) over the ambience, exactly when the live alarm would.

It also writes poster.jpg and chapters.json for the static site. The videos live only
there; samples/ holds the source clips. Phone detection is off, as in evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

from . import hud
from .alerts import CUES, VoicePolicy
from .detector import DEFAULT_MODEL, FaceTracker
from .evaluate import DEFAULT_CONFIG
from .state import Config, DriverState, Frame
from .voice import clip_for

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "web" / "demo"

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


def _thumbs(clip: Path) -> list[np.ndarray]:
    """Small grey frames of the driver's half of the picture, for matching cut points."""
    cap, out = cv2.VideoCapture(str(clip)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (128, 72), interpolation=cv2.INTER_AREA)
        out.append(g[7:, 38:96].astype(np.float32))  # the driver, not the road racing past the windows
    cap.release()
    return out


def best_cuts(clips: list[Path], min_ends: list[float], head: float = 1.5) -> list[tuple[float, float]]:
    """(start, end) per clip so that every join, and the loop back to the start, falls where
    the outgoing and incoming frames look most alike: the driver in the same pose.

    Each clip must run to at least its min_end (so its scenario plays out) and may skip up
    to `head` seconds of its opening, which is the shared reference pose anyway.
    """
    thumbs = [_thumbs(c) for c in clips]
    rates = [cv2.VideoCapture(str(c)).get(cv2.CAP_PROP_FPS) or 24.0 for c in clips]
    n = len(clips)
    starts, ends = [0] * n, [len(t) - 1 for t in thumbs]
    # Choose each join's (end of i, start of i+1) pair; the last join loops to clip 0.
    for i in range(n):
        j = (i + 1) % n
        a, b = thumbs[i], thumbs[j]
        lo = min(int(min_ends[i] * rates[i]), len(a) - 1)
        heads = range(0, min(int(head * rates[j]), len(b) - 1) + 1)
        best = min((float(np.abs(a[e] - b[s]).mean()), e, s) for e in range(lo, len(a)) for s in heads)
        ends[i], starts[j] = best[1], best[2]
    return [(s / r, (e + 1) / r) for s, e, r in zip(starts, ends, rates)]


def join(clips: list[Path], out: Path, cuts: list[tuple[float, float]], ambience: Path | None = None,
         xfade: float = 0.6) -> list[float]:
    """Join the clips into one continuous, seamlessly looping take; returns where each scenario starts.

    Picture: the clips' own sound is dropped, the picture is mirrored (driver on the left of
    the car), and each clip is trimmed to `cuts` and dissolved into the next. The last clip
    dissolves into the opening of the first, and the reel starts just after that opening,
    so playing it on loop never shows a cut.
    Sound: one steady cabin ambience for the whole reel (a synthesised road rumble and tyre
    hiss, or `ambience` looped), with no fades, so it loops as cleanly as the picture.
    """
    n = len(clips)
    (s0, e0), x = cuts[0], xfade
    # Segments: clip 0 from just after its opening, the other clips, then clip 0's opening again.
    # (input, start, end); clip 0 is opened a second time (input n) for the closing segment.
    segs = [(0, s0 + x, e0)] + [(i, *cuts[i]) for i in range(1, n)] + [(n, s0, s0 + x + 0.1)]
    lens = [e - s for _, s, e in segs]
    total = sum(lens[:-1]) - x * (n - 1)

    inputs, chains = [], []
    for clip in [*clips, clips[0]]:
        inputs += ["-an", "-i", str(clip)]
    for k, (i, s, e) in enumerate(segs):
        chains.append(f"[{i}:v]trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS,"
                      f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:-1:-1,hflip,"
                      f"fps={FPS},setsar=1,format=yuv420p[v{k}]")
    last, offset, starts = "[v0]", 0.0, [0.0]
    for k in range(1, len(segs)):
        offset += lens[k - 1] - x
        chains.append(f"{last}[v{k}]xfade=transition=fade:duration={x}:offset={offset:.3f}[x{k}]")
        last = f"[x{k}]"
        if k < n:
            starts.append(round(offset + x / 2, 3))
    chains.append(f"{last}trim=duration={total:.3f},setpts=PTS-STARTPTS[v]")

    a = n + 1
    if ambience is None:
        # Two decorrelated channels of cabin noise: low road rumble plus a quieter tyre hiss.
        for ch, seed in (("L", 7), ("R", 11)):
            chains.append(f"anoisesrc=color=brown:amplitude=0.9:seed={seed}:sample_rate={RATE}:duration={total + 1:.3f},"
                          f"lowpass=f=220,lowpass=f=220[rum{ch}]")
            chains.append(f"anoisesrc=color=pink:amplitude=0.25:seed={seed + 100}:sample_rate={RATE}:duration={total + 1:.3f},"
                          f"highpass=f=400,lowpass=f=3000,volume=0.22[his{ch}]")
            chains.append(f"[rum{ch}][his{ch}]amix=inputs=2:normalize=0[m{ch}]")
        chains.append("[mL][mR]join=inputs=2:channel_layout=stereo[bed0]")
    else:
        inputs += ["-stream_loop", "-1", "-i", str(ambience)]
        chains.append(f"[{a}:a]aformat=sample_rates={RATE}:channel_layouts=stereo[bed0]")
    chains.append(f"[bed0]atrim=duration={total:.3f},asetpts=PTS-STARTPTS,loudnorm=I=-30:LRA=2:TP=-6,aresample={RATE}[a]")

    ffmpeg(*inputs, "-filter_complex", ";".join(chains), "-map", "[v]", "-map", "[a]", "-t", f"{total:.3f}",
           "-c:v", "libx264", "-crf", "22", "-preset", "slow", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out))
    return starts


FATIGUE_NOTE = "Demo: three yawns in seconds. In real driving the window is 5 minutes."


def annotate(reel: Path, out: Path, cfg: Config, model: Path) -> list[tuple[float, str]]:
    """Run the monitor over the reel, drawing the HUD. Returns (t, phrase) for every spoken alert."""
    cap = cv2.VideoCapture(str(reel))
    rate = cap.get(cv2.CAP_PROP_FPS) or FPS
    writer = None
    state, voice, said = DriverState(cfg), VoicePolicy(), []
    with FaceTracker(model) as tracker:  # one drive: calibrated once on the opener, never reset at a cut
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
            s = state.update(f)
            cue = voice.update(s.reasons, t, s.yawns)
            if cue is not None:
                said.append((t, cue.phrase))
            note = FATIGUE_NOTE if any(r.startswith("Fatigue: 3 yawns") for r in s.reasons) else None
            hud.draw(frame, face, s, rate, None, cfg, t, note=note)
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


def mix(video: Path, bed: Path | None, said: list[tuple[float, str]], voices: dict[str, Path], out: Path) -> list[tuple[float, str]]:
    """H.264 mux of the annotated video with the bed's sound (ducked, if any) plus each phrase at its time."""
    inputs = ["-i", str(video)]
    chains, labels, placed, free_at = [], [], [], 0.0
    if bed is not None:
        inputs += ["-i", str(bed)]
        chains.append("[1:a]volume=0.7[bed]")
        labels.append("[bed]")
    first = len(inputs) // 2
    for i, (t, phrase) in enumerate(said):
        at = max(t, free_at)  # one voice: never talk over the previous phrase
        free_at = at + wav_seconds(voices[phrase]) + 0.2
        placed.append((round(at, 2), phrase))
        inputs += ["-i", str(voices[phrase])]
        ms = int(at * 1000)
        chains.append(f"[{first + i}:a]aformat=sample_rates={RATE}:channel_layouts=stereo,"
                      f"adelay={ms}|{ms},volume=1.8[s{i}]")
        labels.append(f"[s{i}]")
    video_args = ["-c:v", "libx264", "-crf", "22", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if not labels:
        ffmpeg(*inputs, "-map", "0:v", *video_args, "-an", str(out))
        return placed
    mixed = labels[0] if len(labels) == 1 else f"{''.join(labels)}amix=inputs={len(labels)}:duration={'first' if bed else 'longest'}:normalize=0"
    chains.append(f"{mixed},alimiter=limit=0.95[a]" if len(labels) > 1 else f"{mixed}alimiter=limit=0.95[a]")
    ffmpeg(*inputs, "-filter_complex", ";".join(chains), "-map", "0:v", "-map", "[a]", *video_args,
           "-c:a", "aac", "-b:a", "160k", "-shortest", str(out))
    return placed


def encode_recording(raw: Path, source_audio: Path | None, said: list[tuple[float, str]], out: Path) -> list[tuple[float, str]]:
    """Turn the HUD recording (mp4v from OpenCV) into H.264 with the spoken alerts mixed in.

    `source_audio` is the original video when the input was a file (its sound becomes the bed).
    """
    with tempfile.TemporaryDirectory() as tmp:
        voices = synthesize({phrase for _, phrase in said}, Path(tmp)) if said else {}
        bed = source_audio if source_audio is not None and probe(source_audio)[1] else None
        return mix(raw, bed, said, voices, out)


def main() -> None:
    p = argparse.ArgumentParser(prog="driver_monitor.reel", description=__doc__.splitlines()[0])
    p.add_argument("spec", type=Path, help="scenarios JSON (clip order = reel order)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where reel.mp4, showcase.mp4, poster.jpg and chapters.json go (default: web/demo)")
    p.add_argument("--ambience", default=None, help="sound file to loop as the ambience (default: synthesised cabin noise)")
    p.add_argument("--xfade", type=float, default=0.6, help="dissolve between scenarios, seconds")
    args = p.parse_args()

    folder = args.spec.parent
    scenarios = json.loads(args.spec.read_text(encoding="utf-8"))
    clips = [folder / sc["clip"] for sc in scenarios]
    args.out.mkdir(parents=True, exist_ok=True)
    reel, showcase = args.out / "reel.mp4", args.out / "showcase.mp4"

    cuts = best_cuts(clips, [sc.get("min_end", 0.0) for sc in scenarios])
    ambience = folder / args.ambience if args.ambience else None
    starts = join(clips, reel, cuts, ambience, args.xfade)
    total, _ = probe(reel)
    print(f"reel.mp4      {total:.1f}s, {len(clips)} scenarios, loops seamlessly, "
          f"ambience: {ambience.name if ambience else 'synthesised cabin noise'}")
    for sc, (s, e) in zip(scenarios, cuts):
        print(f"  {sc['clip']:14} {s:4.2f}-{e:5.2f}s")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        said = annotate(reel, tmp / "hud.mp4", Config.load(args.config), args.model)
        voices = synthesize({phrase for _, phrase in said}, tmp)
        placed = mix(tmp / "hud.mp4", reel, said, voices, showcase)
    print(f"showcase.mp4  {len(placed)} spoken alerts")
    for t, phrase in placed:
        print(f"  {t:5.1f}s  {phrase}")

    ffmpeg("-ss", "1.5", "-i", str(reel), "-frames:v", "1", "-q:v", "3", str(args.out / "poster.jpg"))
    ends = starts[1:] + [total]
    chapters = [{"label": sc["label"], "start": round(s, 2), "end": round(e, 2), "expect": sc.get("summary", "")}
                for sc, s, e in zip(scenarios, starts, ends, strict=True)]
    (args.out / "chapters.json").write_text(
        "[\n" + ",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in chapters) + "\n]\n", encoding="utf-8")
    print(f"wrote {args.out}: reel.mp4, showcase.mp4, poster.jpg, chapters.json")


if __name__ == "__main__":
    main()

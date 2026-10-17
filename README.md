# Driver Monitor

Real-time driver drowsiness and distraction detection from a single webcam, built with Python, OpenCV and MediaPipe.

It tracks 478 facial landmarks per frame, turns them into a few interpretable signals, and reasons about them over time to separate a normal blink from a microsleep, or a mirror check from eyes off the road.

## What it detects

| Condition | Signal | Rule (defaults) | Level |
|---|---|---|---|
| Microsleep | Eye Aspect Ratio (EAR) | Eyes closed ≥ 1.2 s | **ALERT** + alarm |
| Eyes off road | Head yaw / pitch | Turned > 28° or looking down > 18° for ≥ 2 s | **ALERT** + alarm |
| Fatigue | PERCLOS | Eyes closed > 15% of the last 60 s | WARN |
| Fatigue | Mouth Aspect Ratio (MAR) | 3 yawns within 5 min | WARN |
| Driver not visible | Face presence | No face for ≥ 2 s | WARN |

### How it works

```
camera ─▶ MediaPipe FaceLandmarker ─▶ per-frame signals ─▶ temporal state machine ─▶ HUD + alarm + log
             (478 landmarks,            EAR, MAR,              durations, sliding
              head transform)           pitch / yaw            windows, calibration
```

- **EAR** (Soukupová & Čech, 2016): ratio of eyelid opening to eye width. It falls towards 0 as the eye closes.
- **MAR**: inner-lip opening divided by mouth width, which spikes during a yawn.
- **Head pose**: Euler angles taken from MediaPipe's facial transformation matrix.
- **PERCLOS**: the percentage of time the eyes are closed, an established fatigue metric. It is time-weighted, so frame drops don't skew it.
- **Per-driver calibration**: for the first 3 seconds the system learns *your* open-eye EAR and neutral head pose. Thresholds are relative to that baseline rather than a fixed value that only suits some faces. Press `c` to recalibrate.

The decision logic in [`state.py`](driver_monitor/state.py) is pure Python with explicit timestamps, so it is unit tested without a camera.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m driver_monitor.download_model
python -m driver_monitor
```

Keys: `q` quits, `c` recalibrates.

### Options

```bash
python -m driver_monitor --source 1                    # another camera
python -m driver_monitor --source drive.mp4            # a recorded video
python -m driver_monitor --source drive.mp4 --headless --log events.jsonl
python -m driver_monitor --record out.mp4              # save the annotated video
python -m driver_monitor --no-alarm --mirror
```

`--log` writes one JSON line per state change, for example `{"t": 42.1, "level": "ALERT", "reasons": ["Eyes closed 1.3s"]}`.

## Tests

```bash
pip install pytest
pytest -q
```

## Project layout

```
driver_monitor/
  metrics.py        EAR, MAR, Euler angles from landmarks
  detector.py       MediaPipe FaceLandmarker wrapper
  state.py          temporal logic, calibration, alert levels
  hud.py            overlay rendering
  alarm.py          non-blocking audible alarm
  __main__.py       CLI
tests/              unit tests for metrics and state logic
```

## Limitations and roadmap

- A single RGB camera struggles at night. An IR camera with IR illumination is the standard fix.
- Heavy sunglasses hide the eyes, so EAR is unreliable. Head pose and yawning still work.
- Roadmap: YOLO-based phone and cigarette detection, gaze from iris landmarks, blendshape-based blink scoring, and a Raspberry Pi / Jetson build.

This is a learning and portfolio project, not a certified safety system.

## License

MIT

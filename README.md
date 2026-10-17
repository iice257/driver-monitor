# Driver Monitor

Real-time driver drowsiness and distraction detection from a single webcam, built with Python, OpenCV and MediaPipe.

It tracks 478 facial landmarks and 52 facial blendshapes per frame, detects phones and hands in parallel, and reasons about all of it over time. That separates a normal blink from a microsleep, talking from yawning, and a mirror check from eyes off the road.

## What it detects

| Condition | Signal | Rule (defaults) | Level |
|---|---|---|---|
| Microsleep | EAR + eyeBlink blendshape | Eyes closed ≥ 1.2 s | **ALERT** + alarm |
| Phone in use | Phone detector (EfficientDet / YOLO) | Phone near the face ≥ 1 s, or in view while looking down ≥ 1 s | **ALERT** + alarm |
| Phone call | Hand landmarks | Hand held at the ear ≥ 2 s | **ALERT** + alarm |
| Eyes off road | Head pose + iris gaze | Head turned > 25°, down > 15°, or eyes turned for ≥ 2 s | **ALERT** + alarm |
| Fatigue | PERCLOS | Eyes closed > 15% of the last 60 s | WARN |
| Yawning | jawOpen blendshape | Jaw held open ≥ 1 s; 3 within 5 min is fatigue | WARN |
| Driver not visible | Face presence | No face for ≥ 2 s | WARN |

Every threshold lives in [`config.toml`](config.toml), so you can tune it for your camera position without touching code.

### How it works

```
                ┌─▶ FaceLandmarker (every frame) ─▶ EAR, blendshapes, head pose, iris gaze ─┐
camera frame ───┤                                                                            ├─▶ temporal state machine ─▶ HUD + alarm + log
                └─▶ background thread: phone detector + HandLandmarker (latest frame only) ─┘    (durations, dropout grace,
                                                                                                   hysteresis, calibration)
```

- **EAR** (Soukupová & Čech, 2016): ratio of eyelid opening to eye width. It falls towards 0 as the eye closes.
- **MAR**: inner-lip opening divided by mouth width, which spikes during a yawn.
- **Blendshapes**: MediaPipe's `eyeBlink*` and `jawOpen` scores are pose-robust, so they back up EAR and drive yawn detection. Talking opens the jaw only briefly, which is why a yawn must be held for at least 1 s.
- **Head pose**: the face's own axes are built from 3D landmarks (across the eyes, forehead to chin), and yaw and pitch are read off its normal, with no ±180° wrap-around. An EMA plus hysteresis keeps the state from flickering at the boundary.
- **Gaze**: the iris position between the eye corners catches eyes-only glances that head pose misses.
- **Phone use**: a COCO object detector finds `cell phone`, and hand landmarks catch a hand held at the ear. Both run on a background thread over the latest frame, so the main loop stays at camera rate. Detections tolerate short dropouts.
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
python -m driver_monitor --phone yolo                  # YOLO phone detector (pip install ultralytics, AGPL-3.0)
python -m driver_monitor --phone off                   # face only, lightest
python -m driver_monitor --config night.toml           # alternative thresholds
```

### Tuning

The HUD shows each live signal with its threshold marker. Watch where your values sit, then adjust `config.toml`:

- Yawns don't register: lower `yawn_jaw` (watch the **Jaw open** bar while yawning).
- False "eyes off road" with a high or offset camera: raise `pitch_down_limit_deg` / `yaw_limit_deg`, or recalibrate with `c` while looking at the road.
- Glasses cause false eye closure: lower `ear_ratio` or raise `blink_score`.

`--log` writes one JSON line per state change, for example `{"t": 42.1, "level": "ALERT", "reasons": ["Eyes closed 1.3s"]}`.

## Tests

```bash
pip install pytest
pytest -q
```

## Project layout

```
driver_monitor/
  metrics.py        EAR, MAR, head pose, iris gaze from landmarks
  detector.py       MediaPipe FaceLandmarker wrapper (landmarks + blendshapes)
  phone.py          phone detector + hand landmarks on a background thread
  state.py          temporal logic, calibration, alert levels
  hud.py            overlay rendering
  alarm.py          non-blocking audible alarm
  __main__.py       CLI
tests/              unit tests for metrics and state logic
```

## Limitations and roadmap

- A single RGB camera struggles at night. An IR camera with IR illumination is the standard fix.
- Heavy sunglasses hide the eyes, so EAR is unreliable. Head pose and yawning still work.
- The stock COCO phone detector misses phones that are mostly hidden by a hand. A YOLO model fine-tuned on in-cabin images would do better.
- Roadmap: a fine-tuned in-cabin detector (phone, cigarette, drinking), seatbelt check, and a Raspberry Pi / Jetson build.

This is a learning and portfolio project, not a certified safety system.

## License

MIT

"""Thin wrapper around the MediaPipe Tasks FaceLandmarker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from . import metrics

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODELS_DIR / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass
class FaceReading:
    points: np.ndarray  # (478, 3) pixel coords, z scaled like x
    ear: float
    mar: float
    pitch: float
    yaw: float
    roll: float
    gaze: float  # -1 .. 1, horizontal iris position
    blendshapes: dict[str, float] = field(default_factory=dict)

    @property
    def blink(self) -> float:
        b = self.blendshapes
        return (b.get("eyeBlinkLeft", 0.0) + b.get("eyeBlinkRight", 0.0)) / 2.0

    @property
    def jaw_open(self) -> float:
        return self.blendshapes.get("jawOpen", 0.0)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return metrics.face_box(self.points)


class FaceTracker:
    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Download it with:\n"
                f"  python -m driver_monitor.download_model"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, t_ms: int) -> FaceReading | None:
        # VIDEO mode requires strictly increasing timestamps.
        t_ms = max(int(t_ms), self._last_ts + 1)
        self._last_ts = t_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, t_ms)
        if not result.face_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in result.face_landmarks[0]])
        pitch, yaw, roll = metrics.head_pose(pts)
        shapes = {}
        if result.face_blendshapes:
            shapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
        gaze = metrics.gaze_ratio(pts)
        return FaceReading(
            points=pts,
            ear=metrics.mean_ear(pts),
            mar=metrics.mouth_aspect_ratio(pts),
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            gaze=0.0 if gaze is None else gaze,
            blendshapes=shapes,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

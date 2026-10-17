"""Thin wrapper around the MediaPipe Tasks FaceLandmarker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from . import metrics

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass
class FaceReading:
    points: np.ndarray  # (478, 3) pixel coords (z is relative depth)
    ear: float
    mar: float
    pitch: float
    yaw: float
    roll: float


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
            output_facial_transformation_matrixes=True,
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
        pitch = yaw = roll = 0.0
        if result.facial_transformation_matrixes:
            pitch, yaw, roll = metrics.euler_from_matrix(result.facial_transformation_matrixes[0])
        return FaceReading(
            points=pts,
            ear=metrics.mean_ear(pts),
            mar=metrics.mouth_aspect_ratio(pts),
            pitch=pitch,
            yaw=yaw,
            roll=roll,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

"""Phone-use evidence: a phone object detector plus hand tracking.

Both models are much heavier than the face mesh, so they run on a background
thread over the most recent frame. The main loop never waits for them; it reads
whatever the latest result is and how old it is.

Backends:
  mediapipe (default)  EfficientDet-Lite2 (COCO "cell phone") + HandLandmarker.
  yolo                 Ultralytics YOLO, if installed (pip install ultralytics).
                       Note that Ultralytics is AGPL-3.0 licensed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .detector import MODELS_DIR

OBJECT_MODEL = MODELS_DIR / "efficientdet_lite2.tflite"
OBJECT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite2/float32/latest/efficientdet_lite2.tflite"
)
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


@dataclass
class PhoneResult:
    t: float = 0.0
    phones: list[tuple[Box, float]] = field(default_factory=list)  # (box, score)
    hands: list[np.ndarray] = field(default_factory=list)  # (21, 2) pixel coords


def hand_at_ear(hand: np.ndarray, face: Box) -> bool:
    """Palm beside the face, at ear height: the posture of holding a phone to the ear."""
    x0, y0, x1, y1 = face
    fw, fh = x1 - x0, y1 - y0
    palm = hand[[0, 5, 9, 13, 17]].mean(axis=0)
    beside = (x0 - 0.6 * fw) < palm[0] < (x0 + 0.25 * fw) or (x1 - 0.25 * fw) < palm[0] < (x1 + 0.6 * fw)
    ear_height = (y0 + 0.15 * fh) < palm[1] < (y1 + 0.15 * fh)
    return bool(beside and ear_height)


def _overlap(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def phone_near_face(phones: list[tuple[Box, float]], face: Box) -> bool:
    x0, y0, x1, y1 = face
    pad = 0.5 * (x1 - x0)
    zone = (x0 - pad, y0, x1 + pad, y1 + pad)
    return any(_overlap(b, zone) > 0.3 for b, _ in phones)


log = logging.getLogger(__name__)


class _MediaPipeBackend:
    def __init__(self, score: float) -> None:
        for path, _url in ((OBJECT_MODEL, OBJECT_MODEL_URL), (HAND_MODEL, HAND_MODEL_URL)):
            if not path.exists():
                raise FileNotFoundError(f"{path.name} missing. Run: python -m driver_monitor.download_model")
        self._objects = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(OBJECT_MODEL)),
                running_mode=vision.RunningMode.IMAGE,
                score_threshold=score,
                category_allowlist=["cell phone"],
                max_results=3,
            )
        )
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
            )
        )

    def __call__(self, frame_bgr: np.ndarray) -> tuple[list, list]:
        h, w = frame_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        phones = []
        for d in self._objects.detect(image).detections:
            bb = d.bounding_box
            phones.append(((bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height),
                           d.categories[0].score))
        hands = [np.array([(lm.x * w, lm.y * h) for lm in hand]) for hand in self._hands.detect(image).hand_landmarks]
        return phones, hands

    def close(self) -> None:
        self._objects.close()
        self._hands.close()


class _YoloBackend:
    PHONE_CLASS = 67  # COCO "cell phone"

    def __init__(self, score: float, weights: str = "yolo11n.pt") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise SystemExit("YOLO backend needs: pip install ultralytics") from e
        self._model = YOLO(weights)
        self._score = score
        self._hands = _MediaPipeBackend(score)  # hands still come from MediaPipe

    def __call__(self, frame_bgr: np.ndarray) -> tuple[list, list]:
        res = self._model.predict(frame_bgr, classes=[self.PHONE_CLASS], conf=self._score, verbose=False)[0]
        phones = [(tuple(map(float, b.xyxy[0])), float(b.conf[0])) for b in res.boxes]
        _, hands = self._hands(frame_bgr)
        return phones, hands

    def close(self) -> None:
        self._hands.close()


class PhoneWatcher:
    """Runs the phone/hand models on a background thread over the latest frame."""

    def __init__(self, backend: str = "mediapipe", score: float = 0.3) -> None:
        self.error: str | None = None   # set once if the backend fails; the watcher then idles
        self._backend = None
        try:
            self._backend = _YoloBackend(score) if backend == "yolo" else _MediaPipeBackend(score)
        except Exception as exc:  # missing model, missing ultralytics, no delegate...
            self._fail(exc)
        self._lock = threading.Lock()
        self._frame: tuple[float, np.ndarray] | None = None
        self._result = PhoneResult()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _fail(self, exc: Exception) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        log.error("Phone detection off: %s", self.error)

    def submit(self, frame_bgr: np.ndarray, t: float) -> None:
        if self.error:
            return
        with self._lock:
            self._frame = (t, frame_bgr.copy())
        self._wake.set()

    def latest(self) -> PhoneResult:
        with self._lock:
            return self._result

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.2):
                continue
            self._wake.clear()
            with self._lock:
                job, self._frame = self._frame, None
            if job is None:
                continue
            t, frame = job
            try:
                phones, hands = self._backend(frame)
            except Exception as exc:
                self._fail(exc)
                continue
            with self._lock:
                self._result = PhoneResult(t=t, phones=phones, hands=hands)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2)
        if self._backend is not None:
            self._backend.close()


class NullWatcher:
    """Stand-in when phone detection is disabled."""

    error: str | None = None

    def submit(self, frame_bgr: np.ndarray, t: float) -> None:
        pass

    def latest(self) -> PhoneResult:
        return PhoneResult(t=time.monotonic())

    def close(self) -> None:
        pass

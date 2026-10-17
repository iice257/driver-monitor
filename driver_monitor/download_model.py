"""Fetch the official MediaPipe face landmarker model into ./models."""

from __future__ import annotations

import urllib.request

from .detector import DEFAULT_MODEL, MODEL_URL


def main() -> None:
    DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, DEFAULT_MODEL)
    print(f"Saved to {DEFAULT_MODEL}")


if __name__ == "__main__":
    main()

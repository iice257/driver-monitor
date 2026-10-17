"""Fetch the official MediaPipe models into ./models."""

from __future__ import annotations

import urllib.request

from .detector import DEFAULT_MODEL, MODEL_URL
from .phone import HAND_MODEL, HAND_MODEL_URL, OBJECT_MODEL, OBJECT_MODEL_URL


def main() -> None:
    for path, url in ((DEFAULT_MODEL, MODEL_URL), (OBJECT_MODEL, OBJECT_MODEL_URL), (HAND_MODEL, HAND_MODEL_URL)):
        if path.exists():
            print(f"{path.name} already present")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()

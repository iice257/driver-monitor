"""Non-blocking audible alarm. Uses winsound on Windows, the terminal bell elsewhere."""

from __future__ import annotations

import sys
import threading
import time


class Alarm:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set(self, on: bool) -> None:
        if on and self.enabled:
            self._active.set()
        else:
            self._active.clear()

    def _beep(self) -> None:
        if sys.platform == "win32":
            import winsound

            winsound.Beep(1800, 180)
            winsound.Beep(1400, 180)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(0.36)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._active.wait(timeout=0.1):
                self._beep()
                time.sleep(0.15)

    def close(self) -> None:
        self._stop.set()
        self._active.clear()

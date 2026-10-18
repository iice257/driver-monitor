"""Non-blocking spoken alarm.

Alerts are spoken on a background thread, so the video loop never waits for the voice,
and VoicePolicy decides what to say and how often. The voice, best first:
  1. the pre-rendered neural voice clips in driver_monitor/voice (see voice.py), played
     offline with winsound on Windows;
  2. system text-to-speech via pyttsx3 (SAPI5 / NSSpeech / eSpeak), a female voice if
     one is installed;
  3. beeps: winsound on Windows, the terminal bell elsewhere.
"""

from __future__ import annotations

import queue
import sys
import threading
import time

from .alerts import Cue, VoicePolicy
from .voice import clip_for


def beep(urgent: bool = True) -> None:
    if sys.platform == "win32":
        import winsound

        winsound.Beep(1800, 180)
        if urgent:
            winsound.Beep(1400, 180)
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()
        time.sleep(0.36)


class Alarm:
    def __init__(self, enabled: bool = True, voice: bool = True, rate: int = 185) -> None:
        self.enabled = enabled
        self.policy = VoicePolicy()
        self._want_voice = voice
        self._rate = rate
        self._queue: queue.Queue[tuple[Cue, bool]] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if enabled:
            self._thread.start()

    def update(self, reasons: list[str], t: float, urgent: bool, yawns: int | None = None) -> Cue | None:
        """Feed the current reasons; queues a phrase when one is due. Returns the cue spoken."""
        cue = self.policy.update(reasons, t, yawns)
        if cue is not None and self.enabled:
            try:
                self._queue.put_nowait((cue, urgent))
            except queue.Full:  # already behind: drop rather than talk about the past
                pass
        return cue

    def reset(self) -> None:
        self.policy.reset()

    def _engine(self):
        if not self._want_voice:
            return None
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            female = [v for v in engine.getProperty("voices")
                      if "female" in str(getattr(v, "gender", "")).lower() or "zira" in v.name.lower()]
            if female:
                engine.setProperty("voice", female[0].id)
            return engine
        except Exception:  # no pyttsx3, or no speech engine on this machine
            return None

    def _play_clip(self, cue: Cue) -> bool:
        clip = clip_for(cue) if self._want_voice and sys.platform == "win32" else None
        if clip is None:
            return False
        import winsound

        try:
            winsound.PlaySound(str(clip), winsound.SND_FILENAME)
            return True
        except RuntimeError:
            return False

    def _run(self) -> None:
        engine = self._engine()  # SAPI objects must live on the thread that uses them
        while not self._stop.is_set():
            try:
                cue, urgent = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._play_clip(cue):
                continue
            if engine is not None:
                try:
                    engine.say(cue.phrase)
                    engine.runAndWait()
                    continue
                except Exception:
                    engine = None
            beep(urgent)

    def close(self) -> None:
        self._stop.set()

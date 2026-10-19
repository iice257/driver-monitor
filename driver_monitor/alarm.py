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

import sys
import threading
import time

from .alerts import AlarmPolicy, Cue
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


def _clip_seconds(path) -> float:
    import wave

    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 2.0


class Alarm:
    def __init__(self, enabled: bool = True, voice: bool = True, rate: int = 185) -> None:
        self.enabled = enabled
        self.policy = AlarmPolicy()
        self._want_voice = voice
        self._rate = rate
        self._lock = threading.Lock()
        self._next: tuple[Cue, bool] | None = None   # newest phrase wins; an ALERT replaces a waiting one
        self._interrupt = threading.Event()          # cut the phrase that is playing now
        self._siren_hz = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if enabled:
            self._thread.start()

    def update(self, reasons: list[str], t: float, urgent: bool, yawns: int | None = None) -> Cue | None:
        """Feed the current reasons; starts a phrase or the siren when one is due. Returns the cue."""
        d = self.policy.update(reasons, t, urgent, yawns)
        if self.enabled:
            with self._lock:
                self._siren_hz = d.siren_hz
                if d.cue is not None and not (self._next and self._next[1] and not d.preempt):
                    self._next = (d.cue, d.preempt)
            if d.preempt:
                self._interrupt.set()
        return d.cue

    def reset(self) -> None:
        self.policy.reset()
        with self._lock:
            self._siren_hz, self._next = 0.0, None
        self._interrupt.set()

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
        """Play the clip; returns False if there is none. Cut short when _interrupt is set."""
        clip = clip_for(cue) if self._want_voice and sys.platform == "win32" else None
        if clip is None:
            return False
        import winsound

        try:
            winsound.PlaySound(str(clip), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except RuntimeError:
            return False
        self._interrupt.wait(_clip_seconds(clip))
        winsound.PlaySound(None, winsound.SND_PURGE)
        return True

    def _siren(self, hz: float) -> None:
        """One cycle of the two-tone alarm: high then low, at full volume."""
        half = max(40, int(500 / hz))
        if sys.platform == "win32":
            import winsound

            winsound.Beep(1900, half)
            winsound.Beep(1250, half)
        else:
            sys.stdout.write("")
            sys.stdout.flush()
            time.sleep(2 * half / 1000)

    def _run(self) -> None:
        engine = self._engine()  # SAPI objects must live on the thread that uses them
        while not self._stop.is_set():
            with self._lock:
                hz, item, self._next = self._siren_hz, self._next, None
            if hz > 0:
                self._siren(hz)
                continue
            if item is None:
                time.sleep(0.05)
                continue
            cue, urgent = item
            self._interrupt.clear()
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
        self._interrupt.set()

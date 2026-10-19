"""What each alert reason is called out loud and on screen, and when it is (re)spoken.

The state machine reports reasons such as "Eyes closed 1.4s" or "Fatigue: 3 yawns".
This module turns them into a short phrase ("Eyes closed. Wake up!") and a headline
for the on-screen banner, and rate-limits the phrases so a held condition repeats at a
steady pace instead of every frame. It is pure Python with explicit timestamps, so the
same policy drives the live alarm, the offline reel renderer and the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cue:
    key: str        # stable id for rate limiting
    headline: str   # on-screen banner text
    phrase: str     # what the voice says
    repeat_s: float  # while the condition holds, say it again after this long


# Matched by prefix against the state machine's reasons, most specific first.
CUES: tuple[tuple[str, Cue], ...] = (
    ("Eyes closed", Cue("eyes", "EYES CLOSED", "Eyes closed, please stay awake.", 4.0)),
    ("Phone call", Cue("call", "PHONE CALL DETECTED", "Phone call detected, please keep your eyes on the road.", 8.0)),
    ("Phone in use", Cue("phone", "PHONE DETECTED", "Phone detected, please put your phone away.", 8.0)),
    ("Eyes off road", Cue("away", "DRIVER DISTRACTED", "Driver distracted, please focus on the road.", 5.0)),
    ("Driver not visible", Cue("absent", "DRIVER NOT VISIBLE", "Driver not visible, please pay attention.", 10.0)),
    ("Fatigue", Cue("fatigue", "FATIGUE DETECTED", "Fatigue detected, please take a break soon.", 30.0)),
    ("Yawn", Cue("yawn", "YAWNING DETECTED", "Yawning detected, please drive carefully.", 30.0)),
)


def cue_for(reason: str) -> Cue | None:
    for prefix, cue in CUES:
        if reason.startswith(prefix):
            return cue
    return None


@dataclass
class VoicePolicy:
    """Decides which phrase, if any, to speak now.

    A cue is spoken when its reason first appears (unless the same cue was spoken less
    than `min_gap_s` ago) and again every `repeat_s` while it holds. At most one phrase
    is returned per update, in the state machine's priority order; a cue that loses out
    stays pending and is spoken on a later update. Passing the running yawn count makes
    every new yawn its own announcement, even while the "Yawn" reason is still showing.
    """

    min_gap_s: float = 2.5
    _last: dict[str, float] = field(default_factory=dict)
    _spoken: set[str] = field(default_factory=set)  # cues already announced in their current episode
    _yawns: int = 0

    def update(self, reasons: list[str], t: float, yawns: int | None = None) -> Cue | None:
        active = [c for c in (cue_for(r) for r in reasons) if c is not None]
        keys = {c.key for c in active}
        self._spoken &= keys  # a cue that ended starts a fresh episode next time
        if yawns is not None:
            if yawns > self._yawns:
                self._spoken.discard("yawn")
            self._yawns = yawns
        for cue in active:
            last = self._last.get(cue.key)
            since = float("inf") if last is None else t - last
            due = since >= cue.repeat_s if cue.key in self._spoken else since >= self.min_gap_s
            if due:
                self._last[cue.key] = t
                self._spoken.add(cue.key)
                return cue
        return None

    def reset(self) -> None:
        self._last.clear()
        self._spoken.clear()
        self._yawns = 0


@dataclass(frozen=True)
class Decision:
    cue: Cue | None = None      # phrase to start now
    preempt: bool = False       # stop whatever is playing first (the cue is an ALERT)
    siren_hz: float = 0.0       # > 0: play the two-tone alarm at this many cycles per second


@dataclass
class AlarmPolicy:
    """VoicePolicy plus the safety rules: ALERT cues pre-empt, closed eyes escalate to a siren.

    The voice stays for WARN-level cues. If the eyes are still closed `siren_after_s` after the
    first phrase, the voice stops and a loud two-tone alarm takes over, getting faster
    (`siren_start_hz` to `siren_max_hz`, +`siren_ramp_hz` per second) until the eyes open.
    """

    siren_after_s: float = 2.0
    siren_start_hz: float = 2.0
    siren_max_hz: float = 6.0
    siren_ramp_hz: float = 0.5
    voice: VoicePolicy = field(default_factory=VoicePolicy)
    _eyes_since: float | None = None

    def update(self, reasons: list[str], t: float, urgent: bool, yawns: int | None = None) -> Decision:
        eyes = any(r.startswith("Eyes closed") for r in reasons)
        if not eyes:
            self._eyes_since = None
        elif self._eyes_since is None:
            self._eyes_since = t
        if self._eyes_since is not None and t - self._eyes_since >= self.siren_after_s:
            hz = min(self.siren_max_hz, self.siren_start_hz + self.siren_ramp_hz * (t - self._eyes_since - self.siren_after_s))
            self.voice.update([r for r in reasons if not r.startswith("Eyes closed")], t, yawns)
            return Decision(siren_hz=hz, preempt=True)
        cue = self.voice.update(reasons, t, yawns)
        if cue is None:
            return Decision()
        return Decision(cue=cue, preempt=urgent)

    def reset(self) -> None:
        self.voice.reset()
        self._eyes_since = None

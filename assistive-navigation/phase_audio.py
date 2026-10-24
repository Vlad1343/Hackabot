"""
Laptop proximity phase tones.
3 phases:
- SAFE: soft/low tone, slow repeats
- WARNING: medium tone, medium repeats
- DANGER: loud/high ("thinner") tone, fast repeats
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Dict

import numpy as np

try:
    import simpleaudio as sa
except Exception:
    sa = None


@dataclass(frozen=True)
class PhaseProfile:
    frequency_hz: float
    duration_sec: float
    volume: float
    repeat_interval_sec: float


PHASE_PROFILES: Dict[str, PhaseProfile] = {
    "SAFE": PhaseProfile(frequency_hz=420.0, duration_sec=0.11, volume=0.16, repeat_interval_sec=1.30),
    "WARNING": PhaseProfile(frequency_hz=900.0, duration_sec=0.10, volume=0.42, repeat_interval_sec=0.70),
    # Strongly distinguish danger on laptops: higher pitch, shorter pulse, much faster cadence.
    "DANGER": PhaseProfile(frequency_hz=2100.0, duration_sec=0.06, volume=0.95, repeat_interval_sec=0.12),
}


def _tone_wave(frequency_hz: float, duration_sec: float, volume: float, sample_rate: int = 44100) -> np.ndarray:
    n = max(1, int(sample_rate * duration_sec))
    t = np.linspace(0.0, duration_sec, n, False, dtype=np.float32)
    # Sine with short attack/release envelope to avoid clicks.
    wave = np.sin(2.0 * math.pi * float(frequency_hz) * t)
    env = np.ones_like(wave)
    edge = max(1, int(0.01 * sample_rate))
    env[:edge] = np.linspace(0.0, 1.0, edge)
    env[-edge:] = np.linspace(1.0, 0.0, edge)
    y = wave * env * float(max(0.0, min(1.0, volume)))
    return (y * 32767.0).astype(np.int16)


class PhaseToneEngine:
    def __init__(self) -> None:
        self._last_play = {"SAFE": 0.0, "WARNING": 0.0, "DANGER": 0.0}
        self._cache = {
            name: _tone_wave(p.frequency_hz, p.duration_sec, p.volume)
            for name, p in PHASE_PROFILES.items()
        }
        self._warned_no_audio = False
        self._backend = self._select_backend()
        self._wav_files: Dict[str, str] = {}
        if self._backend == "afplay":
            self._prepare_wav_files()

    def _select_backend(self) -> str:
        # On macOS, prefer afplay to avoid simpleaudio segfaults on some builds.
        if os.name == "posix" and shutil.which("afplay"):
            return "afplay"
        if sa is not None:
            return "simpleaudio"
        return "none"

    def _prepare_wav_files(self) -> None:
        for phase, pcm in self._cache.items():
            fd, path = tempfile.mkstemp(prefix=f"hackabot_{phase.lower()}_", suffix=".wav")
            os.close(fd)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(44100)
                wf.writeframes(pcm.tobytes())
            self._wav_files[phase] = path

    def play_if_due(self, phase: str) -> None:
        p = PHASE_PROFILES.get(phase)
        if p is None:
            return
        now = time.monotonic()
        if now - self._last_play[phase] < p.repeat_interval_sec:
            return
        self._last_play[phase] = now
        self._play_now(phase)

    def play_force(self, phase: str) -> None:
        """Play immediately, bypassing repeat interval checks."""
        if phase not in PHASE_PROFILES:
            return
        self._last_play[phase] = time.monotonic()
        self._play_now(phase)

    def _play_now(self, phase: str) -> None:
        p = PHASE_PROFILES.get(phase)
        if p is None:
            return

        if self._backend == "afplay":
            path = self._wav_files.get(phase)
            if not path:
                return
            try:
                subprocess.Popen(
                    ["afplay", path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:
                print(f"[PhaseTone] afplay error: {exc}")
            return

        if self._backend == "simpleaudio" and sa is not None:
            try:
                sa.play_buffer(self._cache[phase], 1, 2, 44100)
                return
            except Exception as exc:
                print(f"[PhaseTone] simpleaudio error: {exc}")

        if self._backend == "none":
            if not self._warned_no_audio:
                print("[PhaseTone] simpleaudio not installed; tone playback disabled")
                self._warned_no_audio = True
            return

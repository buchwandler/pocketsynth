from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import as_float32_mono
from .errors import VoicePromptError


@dataclass(frozen=True, slots=True)
class PreparedVoice:
    state: Any
    sample_rate: int
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_pcm_wav(path: str | Path) -> tuple[np.ndarray, int]:
    source = Path(path)
    with wave.open(str(source), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or width != 2:
        raise VoicePromptError("MVP WAV voice prompts must be mono 16-bit PCM")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    audio = as_float32_mono(audio)
    if source_rate == target_rate or not audio.size:
        return audio
    target_length = max(1, round(audio.size * target_rate / source_rate))
    old_x = np.linspace(0.0, 1.0, audio.size, endpoint=True)
    new_x = np.linspace(0.0, 1.0, target_length, endpoint=True)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def prepare_voice(runtime: Any, source: str | Path | tuple[np.ndarray, int] | PreparedVoice, *, sample_rate: int) -> PreparedVoice:
    if isinstance(source, PreparedVoice):
        return source
    if isinstance(source, tuple):
        audio, source_rate = source
        label = None
    else:
        audio, source_rate = _read_pcm_wav(source)
        label = str(source)
    audio = _resample_linear(audio, int(source_rate), sample_rate)
    method = getattr(runtime, "prepare_voice", None)
    if method is None:
        raise VoicePromptError(
            "OnnxVoice PocketAdapter must provide prepare_voice(audio, sample_rate=...)"
        )
    state = method(audio, sample_rate=sample_rate)
    return PreparedVoice(state=state, sample_rate=sample_rate, source=label)

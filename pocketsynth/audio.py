from __future__ import annotations

import wave
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .errors import ModelInferenceError


def as_float32_mono(audio: np.ndarray, *, name: str = "audio") -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1:
        raise ModelInferenceError(f"{name} must be one-dimensional, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ModelInferenceError(f"{name} contains non-finite samples")
    return value


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(as_float32_mono(audio), -1.0, 1.0) * 32767.0).astype(np.int16)


def write_wav(target: str | Path | BinaryIO, audio: np.ndarray, sample_rate: int) -> None:
    destination = str(target) if isinstance(target, Path) else target
    pcm = float_to_int16(audio)
    with wave.open(destination, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

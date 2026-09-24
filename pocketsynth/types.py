from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .audio import float_to_int16, write_wav
from .diagnostics import RuntimeDiagnostics, SynthesisTiming
from .errors import ModelInferenceError, OptionalDependencyError


@dataclass(frozen=True, slots=True)
class SynthesisSegment:
    id: str
    text: str
    language: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("segment id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("segment text must be a string")
        if self.language is not None and not isinstance(self.language, str):
            raise TypeError("segment language must be a string or None")


@dataclass(slots=True)
class RenderedChunk:
    index: int
    text: str
    model_text: str
    token_ids: tuple[int, ...]
    audio: np.ndarray
    sample_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if self.audio.ndim != 1 or not np.all(np.isfinite(self.audio)):
            raise ModelInferenceError("chunk audio must be one-dimensional and finite")
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise ModelInferenceError("chunk sample_rate must be a positive integer")
        if self.index < 0:
            raise ValueError("chunk index must be >= 0")
        self.token_ids = tuple(self.token_ids)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class RenderedSegment:
    id: str
    audio: np.ndarray
    sample_rate: int
    text: str
    language: str
    token_ids: tuple[int, ...]
    chunks: tuple[RenderedChunk, ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: RuntimeDiagnostics | None = None
    timing: SynthesisTiming | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("result id must be a non-empty string")
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if self.audio.ndim != 1 or not np.all(np.isfinite(self.audio)):
            raise ModelInferenceError("result audio must be one-dimensional and finite")
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise ModelInferenceError("result sample_rate must be a positive integer")
        self.token_ids = tuple(self.token_ids)
        self.chunks = tuple(self.chunks)
        self.warnings = tuple(self.warnings)
        self.metadata = dict(self.metadata)

    @property
    def duration_seconds(self) -> float:
        return self.audio.size / self.sample_rate

    @property
    def int16_bytes(self) -> bytes:
        return float_to_int16(self.audio).tobytes()

    def save_wav(self, target: str | Path | BinaryIO) -> str | Path | BinaryIO:
        """Write mono PCM16 WAV audio, clipping samples during conversion."""
        if isinstance(target, (str, Path)):
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_wav(path, self.audio, self.sample_rate)
            return path
        write_wav(target, self.audio, self.sample_rate)
        return target

    def play(self, *, wait: bool = True) -> None:
        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "Audio playback requires sounddevice. Install pocketsynth[playback]."
            ) from exc
        sd.play(self.audio, self.sample_rate, blocking=wait)

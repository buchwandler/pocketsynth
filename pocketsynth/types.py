from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
from utterplan import UtterancePlan

from .audio import float_to_int16, write_wav
from .diagnostics import RuntimeDiagnostics, TimingDiagnostics
from .errors import ModelInferenceError, OptionalDependencyError


@dataclass(frozen=True, slots=True)
class AudioMarker:
    """A marker in synthesized audio, either resolved or unresolved."""

    id: str
    status: Literal["resolved", "unresolved"]
    sample_offset: int | None = None
    seconds: float | None = None
    plan_unit_id: str | None = None
    reason: str | None = None


@dataclass(slots=True)
class AudioChunk:
    sample_rate: int
    audio: np.ndarray
    token_ids: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if self.audio.ndim != 1 or not np.all(np.isfinite(self.audio)):
            raise ModelInferenceError("audio must be one-dimensional and finite")
        self.token_ids = tuple(self.token_ids)
        self.warnings = tuple(self.warnings)
        self.metadata = dict(self.metadata)

    @property
    def duration_seconds(self) -> float:
        return self.audio.size / self.sample_rate

    @property
    def int16_bytes(self) -> bytes:
        return float_to_int16(self.audio).tobytes()


@dataclass(slots=True)
class AudioResult:
    audio: np.ndarray
    sample_rate: int
    source_text: str
    prepared_text: str
    plan: UtterancePlan | None = None
    plan_id: str | None = None
    chunks: list[AudioChunk] = field(default_factory=list)
    markers: list[AudioMarker] = field(default_factory=list)
    warnings: tuple[str, ...] = ()
    diagnostics: RuntimeDiagnostics | None = None
    timing: TimingDiagnostics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if self.audio.ndim != 1 or not np.all(np.isfinite(self.audio)):
            raise ModelInferenceError("result audio must be one-dimensional and finite")

    @property
    def duration_seconds(self) -> float:
        return self.audio.size / self.sample_rate

    def save_wav(self, target: str | Path | BinaryIO) -> str | Path | BinaryIO:
        """Write audio to a WAV file. Creates parent directories for path targets."""
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


@dataclass(frozen=True, slots=True)
class AudioUnitDescriptor:
    index: int
    unit_kind: Literal["sentence", "paragraph"]
    text: str
    char_start: int | None = None
    char_end: int | None = None
    plan_unit_id: str | None = None
    content_hash: str | None = None
    segment_ids: tuple[str, ...] = ()
    marker_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class AudioUnitResult:
    descriptor: AudioUnitDescriptor
    audio: np.ndarray
    sample_rate: int
    token_ids: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if self.audio.ndim != 1 or not np.all(np.isfinite(self.audio)):
            raise ModelInferenceError("unit audio must be one-dimensional and finite")

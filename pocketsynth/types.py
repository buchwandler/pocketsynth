from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np

from .audio import float_to_int16, write_wav
from .diagnostics import RuntimeDiagnostics, SynthesisTiming
from .errors import InvalidRequestError, ModelInferenceError, OptionalDependencyError


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


def _validate_span(start: int, end: int, text_length: int | None = None) -> None:
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
        or (text_length is not None and end > text_length)
    ):
        limit = "" if text_length is None else f" and end <= {text_length}"
        raise InvalidRequestError(f"span must satisfy 0 <= start < end{limit}")


@dataclass(frozen=True, slots=True)
class LinguisticToken:
    """Source-aligned linguistic context for engines that support it."""

    start: int
    end: int
    text: str | None = None
    pos: str | None = None
    tag: str | None = None
    lemma: str | None = None
    language: str | None = None
    morph: str | None = None

    def __post_init__(self) -> None:
        _validate_span(self.start, self.end)
        for name in ("text", "pos", "tag", "lemma", "language", "morph"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise InvalidRequestError(f"{name} must be a string or None")


@dataclass(frozen=True, slots=True)
class PronunciationOverride:
    """Source-aligned pronunciation instructions for engines that support them."""

    start: int
    end: int
    phonemes: str | None = None
    language: str | None = None
    stress: str | None = None

    def __post_init__(self) -> None:
        _validate_span(self.start, self.end)
        values = (self.phonemes, self.language, self.stress)
        if all(value is None for value in values):
            raise InvalidRequestError("pronunciation override must specify an effect")
        for name, value in zip(("phonemes", "language", "stress"), values, strict=True):
            if value is not None and (not isinstance(value, str) or not value):
                raise InvalidRequestError(f"{name} must be a non-empty string or None")
        if self.stress is not None and self.stress not in {"-2", "-1", "1", "2"}:
            raise InvalidRequestError("stress must be one of '-2', '-1', '1', or '2'")


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """One source-text request and its optional linguistic context."""

    id: str
    text: str
    language: str | None = None
    tokens: tuple[LinguisticToken, ...] = ()
    pronunciation_overrides: tuple[PronunciationOverride, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise InvalidRequestError("request id must be a non-empty string")
        if not isinstance(self.text, str):
            raise InvalidRequestError("request text must be a string")
        if self.language is not None and not isinstance(self.language, str):
            raise InvalidRequestError("request language must be a string or None")
        tokens = tuple(self.tokens)
        overrides = tuple(self.pronunciation_overrides)
        if any(not isinstance(token, LinguisticToken) for token in tokens):
            raise InvalidRequestError("tokens must contain LinguisticToken values")
        if any(not isinstance(item, PronunciationOverride) for item in overrides):
            raise InvalidRequestError(
                "pronunciation_overrides must contain PronunciationOverride values"
            )
        for token in tokens:
            _validate_span(token.start, token.end, len(self.text))
            if token.text is not None and self.text[token.start : token.end] != token.text:
                raise InvalidRequestError("linguistic token text must match its source span")
        for override in overrides:
            _validate_span(override.start, override.end, len(self.text))
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "pronunciation_overrides", overrides)


@dataclass(frozen=True, slots=True)
class WordTiming:
    """Timing span for one source-text word, when an engine provides timestamps."""

    text: str
    char_start: int
    char_end: int
    start_sample: int
    end_sample: int
    segment_id: str
    source: Literal["model_pred_dur"] = "model_pred_dur"

    def start_seconds(self, sample_rate: int) -> float:
        return self.start_sample / sample_rate

    def end_seconds(self, sample_rate: int) -> float:
        return self.end_sample / sample_rate


@dataclass(slots=True)
class SynthesisResult:
    """One atomic Pocket inference result with no public chunk collection."""

    id: str
    audio: np.ndarray
    sample_rate: int
    text: str
    language: str
    warnings: tuple[str, ...] = ()
    word_timings: tuple[WordTiming, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("result id must be a non-empty string")
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if self.audio.ndim != 1 or not np.all(np.isfinite(self.audio)):
            raise ModelInferenceError("result audio must be one-dimensional and finite")
        if self.audio.size == 0:
            raise ModelInferenceError("result audio must not be empty")
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise ModelInferenceError("result sample_rate must be a positive integer")
        self.warnings = tuple(self.warnings)
        self.word_timings = tuple(self.word_timings)
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

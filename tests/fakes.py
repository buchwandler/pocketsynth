"""Fake runtime and shared test fixtures for PocketSynth tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pocketsynth.diagnostics import RuntimeDiagnostics
from pocketsynth.voice import PreparedVoice


@dataclass
class FakePocketRuntime:
    """In-memory fake Pocket runtime for testing PocketPipeline without ONNX."""

    sample_rate: int = 24_000
    bundle_id: str = "fake-pocket"
    precision: str = "int8"
    source_revision: str = "test"
    metadata: Any = None
    _closed: bool = field(default=False, init=False)
    _infer_count: int = field(default=0, init=False)
    _prepare_count: int = field(default=0, init=False)

    def prepare_voice(self, source: Any, *, sample_rate: int | None = None) -> PreparedVoice:
        """Return a deterministic fake PreparedVoice."""
        self._prepare_count += 1
        if isinstance(source, PreparedVoice):
            return source
        return PreparedVoice(
            state={"fake": True},
            sample_rate=sample_rate or self.sample_rate,
            source=str(source) if source else None,
            bundle_id=self.bundle_id,
        )

    def infer(
        self,
        token_ids: list[int],
        voice_state: Any,
        temperature: float = 0.7,
        lsd_steps: int = 1,
        max_frames: int | None = None,
        frames_after_eos: int | None = None,
    ) -> Any:
        """Return deterministic fake audio based on token count."""
        self._infer_count += 1
        # Generate 100 samples per token
        n_samples = len(token_ids) * 100
        t = np.linspace(0, 1, n_samples, dtype=np.float32)
        # Simple sine wave
        audio = np.sin(2 * np.pi * 440 * t) * 0.5

        @dataclass
        class FakeResult:
            audio: np.ndarray
            sample_rate: int

        return FakeResult(audio=audio, sample_rate=self.sample_rate)

    @property
    def diagnostics(self) -> RuntimeDiagnostics:
        return RuntimeDiagnostics(
            bundle_id=self.bundle_id,
            bundle_path="/fake/bundle",
            sample_rate=self.sample_rate,
            precision=self.precision,
            source_revision=self.source_revision,
        )

    def close(self) -> None:
        self._closed = True


@dataclass
class FakeBundleMetadata:
    """Minimal bundle metadata for testing."""

    path: Any = None
    bundle_name: str = "fake-pocket"
    language: str = "en"
    schema_version: int = 2
    sample_rate: int = 24_000
    samples_per_frame: int = 1920
    max_token_per_chunk: int = 50
    tokenizer_file: str = "tokenizer.model"
    bos_before_voice_file: str = "bos_before_voice.npy"
    remove_semicolons: bool = False
    pad_with_spaces_for_short_inputs: bool = False
    model_recommended_frames_after_eos: int | None = None
    predefined_voices: tuple[str, ...] = ()
    raw: dict[str, Any] | None = None


class FakeProcessor:
    """Fake SentencePiece processor for testing."""

    def EncodeAsIds(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def make_test_voice(sample_rate: int = 24_000) -> PreparedVoice:
    """Create a test PreparedVoice with fake state."""
    return PreparedVoice(
        state={"test": True},
        sample_rate=sample_rate,
        source="test.wav",
    )

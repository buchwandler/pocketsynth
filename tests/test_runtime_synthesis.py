from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pocketsynth.config import GenerationConfig
from pocketsynth.errors import (
    EmptyTextError,
    InvalidLanguageError,
    ModelInferenceError,
    RuntimeClosedError,
)
from pocketsynth.runtime import PocketRuntime
from pocketsynth.types import SynthesisRequest, SynthesisSegment
from pocketsynth.voice import PreparedVoice
from tests.fakes import FakeBundleMetadata


class FakeFrontend:
    def split_for_model(self, text: str) -> tuple[str, ...]:
        return tuple(part.strip() for part in text.split("|") if part.strip())

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text if not char.isspace())


class FakeInferenceRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], dict[str, object]]] = []
        self.closed = False

    def infer(self, token_ids: list[int], **kwargs: object) -> SimpleNamespace:
        self.calls.append((token_ids, kwargs))
        return SimpleNamespace(
            audio=np.asarray(token_ids, dtype=np.float32),
            sample_rate=24_000,
        )

    def close(self) -> None:
        self.closed = True


def make_runtime() -> tuple[PocketRuntime, FakeInferenceRuntime]:
    metadata = FakeBundleMetadata(
        language="english_2026-04",
        model_recommended_frames_after_eos=9,
    )
    backend = FakeInferenceRuntime()
    with patch("pocketsynth.runtime.PocketFrontend", return_value=FakeFrontend()):
        runtime = PocketRuntime(
            paths=None,
            metadata=metadata,  # type: ignore[arg-type]
            tokenizer_path=MagicMock(),
            runtime=backend,
            bundle_id="english_2026-04",
            precision="int8",
        )
    return runtime, backend


def make_voice() -> PreparedVoice:
    return PreparedVoice(state=object(), sample_rate=24_000, bundle_id="english_2026-04")


def test_synthesize_renders_one_atomic_request_and_propagates_config() -> None:
    runtime, backend = make_runtime()
    config = GenerationConfig(
        temperature=1.2,
        lsd_steps=3,
        max_frames=100,
        frames_after_eos=4,
    )
    voice = make_voice()

    result = runtime.synthesize(
        SynthesisRequest(id="line-007", text="hello", language="EN-us"),
        voice=voice,
        config=config,
    )

    assert len(backend.calls) == 1
    token_ids, kwargs = backend.calls[0]
    assert result.id == "line-007"
    assert result.text == "hello"
    assert result.language == "en"
    assert result.metadata["token_count"] == len(token_ids)
    assert not hasattr(result, "chunks")
    np.testing.assert_array_equal(result.audio, np.asarray(token_ids, dtype=np.float32))
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 24_000
    assert kwargs == {
        "voice_state": voice.state,
        "temperature": 1.2,
        "lsd_steps": 3,
        "max_frames": 100,
        "frames_after_eos": 4,
    }


def test_iter_chunks_yields_model_chunks_with_request_local_indexes() -> None:
    runtime, backend = make_runtime()
    chunks = list(
        runtime.iter_chunks(
            SynthesisSegment(id="stream", text="one|two"),
            voice=make_voice(),
        )
    )

    assert [chunk.index for chunk in chunks] == [0, 1]
    assert [chunk.text for chunk in chunks] == ["one", "two"]
    assert len(backend.calls) == 2


def test_empty_or_whitespace_text_is_rejected_before_inference() -> None:
    runtime, backend = make_runtime()

    with pytest.raises(EmptyTextError, match="empty or whitespace"):
        runtime.synthesize(
            SynthesisRequest(id="empty", text="  \t "),
            voice=make_voice(),
        )

    assert backend.calls == []


def test_inference_rejects_nonfinite_audio() -> None:
    runtime, _ = make_runtime()

    def invalid_infer(token_ids: list[int], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(audio=np.array([np.nan]), sample_rate=24_000)

    runtime.runtime.infer = invalid_infer
    with pytest.raises(ModelInferenceError, match="finite"):
        runtime.infer_tokens((1,), make_voice(), GenerationConfig())


def test_frames_after_eos_uses_bundle_default_only_when_unspecified() -> None:
    runtime, backend = make_runtime()
    voice = make_voice()
    runtime.infer_tokens((1,), voice, GenerationConfig())
    runtime.infer_tokens((2,), voice, GenerationConfig(frames_after_eos=0))

    assert backend.calls[0][1]["frames_after_eos"] == 9
    assert backend.calls[1][1]["frames_after_eos"] == 0


def test_incompatible_language_fails_before_inference() -> None:
    runtime, backend = make_runtime()

    with pytest.raises(InvalidLanguageError, match="incompatible"):
        runtime.synthesize(
            SynthesisRequest(id="line", text="bonjour", language="fr"),
            voice=make_voice(),
        )

    assert backend.calls == []


def test_synthesize_text_prepares_voice_and_delegates() -> None:
    runtime, backend = make_runtime()
    prepared = make_voice()
    with patch.object(runtime, "prepare_voice", return_value=prepared) as prepare:
        result = runtime.synthesize_text("hello", voice="alba", id="speech-2")

    prepare.assert_called_once_with("alba")
    assert result.id == "speech-2"
    assert len(backend.calls) == 1


def test_runtime_lifecycle_raises_runtime_closed_error() -> None:
    runtime, backend = make_runtime()
    runtime.close()
    runtime.close()

    assert backend.closed
    with pytest.raises(RuntimeClosedError, match="closed"):
        runtime.infer_tokens((1,), make_voice(), GenerationConfig())


def test_from_pretrained_delegates_install_and_resolved_open() -> None:
    resolved = object()
    opened = make_runtime()[0]
    progress = MagicMock()
    with (
        patch("pocketsynth.runtime.install_pretrained_bundle", return_value=resolved) as install,
        patch.object(PocketRuntime, "from_resolved", return_value=opened) as open_resolved,
    ):
        result = PocketRuntime.from_pretrained(
            "english_2026-04",
            precision="fp32",
            cache_dir="cache",
            offline=True,
            refresh_catalog=True,
            force_download=True,
            providers="CPUExecutionProvider",
            provider_options={"device_id": 0},
            session_options="options",
            progress=progress,
        )

    assert result is opened
    install.assert_called_once_with(
        "english_2026-04",
        precision="fp32",
        cache_dir="cache",
        offline=True,
        refresh_catalog=True,
        force_download=True,
        progress=progress,
    )
    open_resolved.assert_called_once_with(
        resolved,
        providers="CPUExecutionProvider",
        provider_options={"device_id": 0},
        session_options="options",
        cache_dir="cache",
        offline=True,
    )

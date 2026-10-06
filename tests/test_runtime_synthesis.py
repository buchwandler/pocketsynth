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

    def prepare_and_encode(self, text: str) -> tuple[str, tuple[int, ...]]:
        model_text = " ".join(text.strip().split())
        return model_text, self.encode_prepared(model_text)

    def encode_prepared(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text if not char.isspace())

    def encode(self, text: str) -> tuple[int, ...]:
        return self.prepare_and_encode(text)[1]


class FakeInferenceRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], dict[str, object]]] = []
        self.closed = False
        self.result_metadata: object = None

    def infer(self, token_ids: list[int], **kwargs: object) -> SimpleNamespace:
        self.calls.append((token_ids, kwargs))
        return SimpleNamespace(
            audio=np.asarray(token_ids, dtype=np.float32),
            sample_rate=24_000,
            metadata=self.result_metadata,
        )

    def close(self) -> None:
        self.closed = True


def make_runtime(
    *, recommendation: int | None = 9, default_temperature: float | None = None
) -> tuple[PocketRuntime, FakeInferenceRuntime]:
    metadata = FakeBundleMetadata(
        language="english_2026-04",
        model_recommended_frames_after_eos=recommendation,
        default_temperature=default_temperature,
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


def test_inference_rejects_empty_audio() -> None:
    runtime, _ = make_runtime()

    def empty_infer(token_ids: list[int], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(audio=np.array([], dtype=np.float32), sample_rate=24_000)

    runtime.runtime.infer = empty_infer
    with pytest.raises(ModelInferenceError, match="must not be empty"):
        runtime.infer_tokens((1,), make_voice(), GenerationConfig())


@pytest.mark.parametrize(
    ("generation", "expected_frames", "expected_source"),
    [
        (GenerationConfig(frames_after_eos=0), 0, "explicit"),
        (GenerationConfig(), 9, "bundle"),
    ],
)
def test_synthesis_eos_precedence_keeps_explicit_zero_and_bundle_recommendation(
    generation: GenerationConfig,
    expected_frames: int,
    expected_source: str,
) -> None:
    runtime, backend = make_runtime()
    result = runtime.synthesize(
        SynthesisRequest(id="precedence", text="Hello, how are you?"),
        voice=make_voice(),
        config=generation,
    )

    assert backend.calls[0][1]["frames_after_eos"] == expected_frames
    assert result.metadata["effective_generation"]["frames_after_eos"] == expected_frames
    assert result.metadata["effective_generation"]["frames_after_eos_source"] == expected_source


@pytest.mark.parametrize(
    ("text", "expected_frames", "expected_source"),
    [
        ("Hello, how are you?", 5, "automatic_short"),
        ("This sentence has more than four words.", 3, "automatic_default"),
    ],
)
def test_synthesis_uses_text_aware_automatic_tail_and_surfaces_metadata(
    text: str,
    expected_frames: int,
    expected_source: str,
) -> None:
    runtime, backend = make_runtime(recommendation=None)
    backend.result_metadata = {
        "frames_generated": 17,
        "eos_detected": True,
        "eos_step": 12,
    }

    result = runtime.synthesize(
        SynthesisRequest(id="automatic", text=text),
        voice=make_voice(),
    )

    assert backend.calls[0][1]["frames_after_eos"] == expected_frames
    assert result.metadata["generation_config"]["frames_after_eos"] is None
    assert result.metadata["effective_generation"] == {
        "temperature": 0.3,
        "temperature_source": "pocketsynth_default",
        "lsd_steps": 1,
        "max_frames": None,
        "frames_after_eos": expected_frames,
        "frames_after_eos_source": expected_source,
    }
    assert result.metadata["backend_inference"] == backend.result_metadata


def test_generation_temperature_prefers_bundle_then_explicit_override() -> None:
    runtime, backend = make_runtime(default_temperature=0.45)
    request = SynthesisRequest(id="recommended", text="Hello")

    recommended = runtime.synthesize(request, voice=make_voice())

    assert backend.calls[0][1]["temperature"] == 0.45
    assert recommended.metadata["effective_generation"]["temperature"] == 0.45
    assert recommended.metadata["effective_generation"]["temperature_source"] == "bundle"

    runtime, backend = make_runtime(default_temperature=0.45)
    explicit = runtime.synthesize(
        request, voice=make_voice(), config=GenerationConfig(temperature=0.8)
    )

    assert backend.calls[0][1]["temperature"] == 0.8
    assert explicit.metadata["effective_generation"]["temperature"] == 0.8
    assert explicit.metadata["effective_generation"]["temperature_source"] == "explicit"


def test_chunked_rendering_resolves_automatic_tail_per_chunk() -> None:
    runtime, backend = make_runtime(recommendation=None)
    chunks = list(
        runtime.iter_chunks(
            SynthesisSegment(
                id="chunked",
                text="This first chunk contains five words|Short",
            ),
            voice=make_voice(),
        )
    )

    assert [call[1]["frames_after_eos"] for call in backend.calls] == [3, 5]
    assert [
        chunk.metadata["effective_generation"]["frames_after_eos_source"] for chunk in chunks
    ] == ["automatic_default", "automatic_short"]


def test_raw_token_inference_preserves_none_without_text_or_bundle_recommendation() -> None:
    runtime, backend = make_runtime(recommendation=None)
    runtime.infer_tokens((1,), make_voice(), GenerationConfig())

    assert backend.calls[0][1]["frames_after_eos"] is None


def test_backend_metadata_rejects_non_scalar_values() -> None:
    runtime, backend = make_runtime(recommendation=None)
    backend.result_metadata = {"recurrent_state": [1, 2, 3]}

    with pytest.raises(ModelInferenceError, match="JSON-safe scalar"):
        runtime.synthesize(
            SynthesisRequest(id="metadata", text="Hello"),
            voice=make_voice(),
        )


def test_incompatible_language_fails_before_inference() -> None:
    runtime, backend = make_runtime()

    with pytest.raises(InvalidLanguageError, match="incompatible"):
        runtime.synthesize(
            SynthesisRequest(id="line", text="bonjour", language="fr"),
            voice=make_voice(),
        )

    assert backend.calls == []


def test_synthesize_text_rejects_language_before_preparing_voice() -> None:
    runtime, backend = make_runtime()
    with patch.object(runtime, "prepare_voice") as prepare:
        with pytest.raises(InvalidLanguageError, match="incompatible"):
            runtime.synthesize_text("bonjour", voice="alba", language="fr")

    prepare.assert_not_called()
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
        progress=progress,
    )

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from pocketsynth.config import GenerationConfig
from pocketsynth.errors import (
    EmptyTextError,
    InvalidGenerationConfigError,
    InvalidLanguageError,
    InvalidVoiceError,
    ModelInferenceError,
    SynthesisInputTooLongError,
    UnsupportedFeatureError,
)
from pocketsynth.runtime import PocketRuntime
from pocketsynth.types import LinguisticToken, PronunciationOverride, SynthesisRequest
from pocketsynth.voice import PreparedVoice
from pocketsynth.voice_level import VoiceLevelConfig
from tests.fakes import FakeBundleMetadata


class RecordingFrontend:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str) -> tuple[int, ...]:
        self.encoded.append(text)
        return tuple(range(len(text)))

    def split_for_model(self, text: str) -> tuple[str, ...]:
        raise AssertionError("strict synthesis must not split for the model")


class RecordingInferenceRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], dict[str, object]]] = []
        self.failure: Exception | None = None

    def infer(self, token_ids: Sequence[int], **kwargs: object) -> SimpleNamespace:
        self.calls.append((tuple(token_ids), kwargs))
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(
            audio=np.asarray(token_ids, dtype=np.float32),
            sample_rate=24_000,
        )


def make_runtime(
    *,
    max_tokens: int = 50,
    raw_metadata: dict[str, object] | None = None,
) -> tuple[PocketRuntime, RecordingFrontend, RecordingInferenceRuntime]:
    metadata = FakeBundleMetadata(
        language="en",
        max_token_per_chunk=max_tokens,
        model_recommended_frames_after_eos=9,
        raw=raw_metadata,
    )
    frontend = RecordingFrontend()
    backend = RecordingInferenceRuntime()
    with patch("pocketsynth.runtime.PocketFrontend", return_value=frontend):
        runtime = PocketRuntime(
            paths=None,
            metadata=metadata,  # type: ignore[arg-type]
            tokenizer_path=None,  # type: ignore[arg-type]
            runtime=backend,
            bundle_id="english-test",
            precision="int8",
            source_revision="revision-1",
        )
    return runtime, frontend, backend


def make_voice(
    *,
    bundle_id: str = "english-test",
    sample_rate: int = 24_000,
    metadata: dict[str, object] | None = None,
    source_revision: str | None = None,
) -> PreparedVoice:
    return PreparedVoice(
        state=object(),
        sample_rate=sample_rate,
        bundle_id=bundle_id,
        fingerprint="voice-fingerprint",
        metadata=metadata or {},
        source_revision=source_revision,
    )


@pytest.fixture(autouse=True)
def no_runtime_diagnostics() -> None:
    with patch("pocketsynth.runtime.runtime_diagnostics", return_value={}):
        yield


def test_strict_synthesis_encodes_full_text_once_and_infers_once() -> None:
    runtime, frontend, backend = make_runtime()
    request = SynthesisRequest(
        id="seg-42",
        text="First sentence. Second sentence.",
        language="EN-us",
    )
    config = GenerationConfig(temperature=1.2, lsd_steps=3, max_frames=100, frames_after_eos=4)

    voice = make_voice()
    result = runtime.synthesize(request, voice=voice, config=config)

    assert frontend.encoded == [request.text]
    assert len(backend.calls) == 1
    assert backend.calls[0][0] == tuple(range(len(request.text)))
    assert result.id == request.id
    assert result.text == request.text
    assert result.language == "en"
    assert result.audio.dtype == np.float32
    assert result.audio.ndim == 1
    assert result.word_timings == ()
    assert not hasattr(result, "chunks")
    assert result.metadata["token_count"] == len(request.text)
    assert result.metadata["bundle_id"] == "english-test"
    assert result.metadata["bundle_revision"] == "revision-1"
    assert result.metadata["generation_config"]["temperature"] == 1.2
    assert backend.calls[0][1] == {
        "voice_state": voice.state,
        "temperature": 1.2,
        "lsd_steps": 3,
        "max_frames": 100,
        "frames_after_eos": 4,
    }


def test_token_count_at_model_limit_succeeds() -> None:
    runtime, frontend, backend = make_runtime(max_tokens=5)

    result = runtime.synthesize(
        SynthesisRequest(id="at-limit", text="hello"),
        voice=make_voice(),
    )

    assert frontend.encoded == ["hello"]
    assert len(backend.calls) == 1
    assert result.metadata["token_count"] == 5


def test_token_count_over_model_limit_fails_before_inference() -> None:
    runtime, frontend, backend = make_runtime(max_tokens=5)

    with pytest.raises(SynthesisInputTooLongError) as caught:
        runtime.synthesize(
            SynthesisRequest(id="too-long", text="hello!"),
            voice=make_voice(),
        )

    assert frontend.encoded == ["hello!"]
    assert backend.calls == []
    assert caught.value.text_length == 6
    assert caught.value.token_count == 6
    assert caught.value.max_tokens == 5
    assert caught.value.bundle_id == "english-test"


def test_empty_and_whitespace_requests_fail_before_voice_preparation_or_encoding() -> None:
    runtime, frontend, backend = make_runtime()
    runtime.prepare_voice = lambda _source: pytest.fail("voice prepared for empty text")  # type: ignore[method-assign]

    for text in ("", " \t\n"):
        with pytest.raises(EmptyTextError):
            runtime.synthesize_text(text, voice="alba")

    assert frontend.encoded == []
    assert backend.calls == []


def test_strict_request_never_calls_sentence_or_model_splitters() -> None:
    runtime, frontend, backend = make_runtime()
    request = SynthesisRequest(id="unsplit", text="One. Two.")
    with (
        patch.object(
            frontend,
            "split_for_model",
            side_effect=AssertionError("model splitting called"),
        ),
        patch(
            "phrasplit.split_sentences",
            side_effect=AssertionError("Phrasplit called"),
        ),
    ):
        runtime.synthesize(request, voice=make_voice())

    assert frontend.encoded == [request.text]
    assert len(backend.calls) == 1


def test_language_and_prepared_voice_are_validated_before_encoding() -> None:
    runtime, frontend, backend = make_runtime()

    with pytest.raises(InvalidLanguageError):
        runtime.synthesize(
            SynthesisRequest(id="wrong-language", text="bonjour", language="fr"),
            voice=make_voice(),
        )
    with pytest.raises(InvalidVoiceError):
        runtime.synthesize(
            SynthesisRequest(id="wrong-voice", text="hello"),
            voice=make_voice(bundle_id="other-bundle"),
        )

    assert frontend.encoded == []
    assert backend.calls == []


def test_unsupported_linguistic_context_is_rejected_before_encoding() -> None:
    runtime, frontend, backend = make_runtime()
    with pytest.raises(UnsupportedFeatureError) as token_error:
        runtime.synthesize(
            SynthesisRequest(
                id="tokens",
                text="hello",
                tokens=(LinguisticToken(start=0, end=5),),
            ),
            voice=make_voice(),
        )
    assert token_error.value.feature == "linguistic_tokens"

    with pytest.raises(UnsupportedFeatureError) as override_error:
        runtime.synthesize(
            SynthesisRequest(
                id="override",
                text="hello",
                pronunciation_overrides=(PronunciationOverride(start=0, end=5, phonemes="həˈloʊ"),),
            ),
            voice=make_voice(),
        )
    assert override_error.value.feature == "pronunciation_overrides"
    assert frontend.encoded == []
    assert backend.calls == []


def test_invalid_generation_config_is_rejected_before_encoding() -> None:
    runtime, frontend, backend = make_runtime()
    with pytest.raises(InvalidGenerationConfigError):
        runtime.synthesize(
            SynthesisRequest(id="bad-config", text="hello"),
            voice=make_voice(),
            config="invalid",  # type: ignore[arg-type]
        )

    assert frontend.encoded == []
    assert backend.calls == []


def test_inference_failures_are_mapped_to_model_inference_error() -> None:
    runtime, frontend, backend = make_runtime()
    backend.failure = RuntimeError("backend failed")

    with pytest.raises(ModelInferenceError, match="backend failed"):
        runtime.synthesize(
            SynthesisRequest(id="failure", text="hello"),
            voice=make_voice(),
        )

    assert frontend.encoded == ["hello"]
    assert len(backend.calls) == 1


def test_infer_tokens_capacity_guard_uses_public_typed_error() -> None:
    runtime, _, backend = make_runtime(max_tokens=2)

    with pytest.raises(SynthesisInputTooLongError) as caught:
        runtime.infer_tokens((1, 2, 3), make_voice(), GenerationConfig(), text_length=8)

    assert caught.value.text_length == 8
    assert caught.value.token_count == 3
    assert caught.value.max_tokens == 2
    assert backend.calls == []


def test_predefined_voice_uses_catalog_gain_and_revision_identity() -> None:
    runtime, _, _ = make_runtime(raw_metadata={"voice_level_calibration": {"alba": -6.0}})
    voice = make_voice(
        metadata={"kind": "predefined", "name": "alba"},
        source_revision="revision-1",
    )

    result = runtime.synthesize(
        SynthesisRequest(id="calibrated", text="hello"),
        voice=voice,
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )

    np.testing.assert_allclose(
        result.audio,
        np.arange(5, dtype=np.float32) * np.float32(10 ** (-6.0 / 20.0)),
    )
    assert result.metadata["voice_identity"] == {
        "kind": "predefined",
        "bundle_id": "english-test",
        "name": "alba",
        "source_revision": "revision-1",
    }
    assert result.metadata["voice_level_application"] == {
        "mode": "calibrated",
        "gain_db": -6.0,
        "source": "catalog",
        "applied": True,
    }


def test_reference_voice_without_catalog_gain_reports_missing_calibration() -> None:
    runtime, _, _ = make_runtime()
    voice = make_voice(metadata={"kind": "reference"})

    result = runtime.synthesize(
        SynthesisRequest(id="reference", text="hello"),
        voice=voice,
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )

    np.testing.assert_array_equal(result.audio, np.arange(5, dtype=np.float32))
    assert result.metadata["voice_identity"] == {
        "kind": "reference",
        "sha256": "voice-fingerprint",
        "sample_rate": 24_000,
    }
    assert result.metadata["voice_level_application"]["source"] == "missing_calibration"


def test_explicit_voice_gain_overrides_calibration_policy() -> None:
    runtime, _, _ = make_runtime(raw_metadata={"voice_level_calibration": {"alba": -6.0}})
    voice = make_voice(
        metadata={"kind": "predefined", "name": "alba"},
        source_revision="revision-1",
    )

    result = runtime.synthesize(
        SynthesisRequest(id="override", text="hello"),
        voice=voice,
        voice_level=VoiceLevelConfig(mode="calibrated", gain_db=3.0),
    )

    np.testing.assert_allclose(
        result.audio,
        np.arange(5, dtype=np.float32) * np.float32(10 ** (3.0 / 20.0)),
    )
    assert result.metadata["voice_level_application"]["source"] == "override"
    assert result.metadata["voice_level_application"]["gain_db"] == 3.0

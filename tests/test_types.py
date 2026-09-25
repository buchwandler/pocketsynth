import numpy as np
import pytest

from pocketsynth.config import GenerationConfig
from pocketsynth.diagnostics import SynthesisTiming
from pocketsynth.errors import (
    InvalidGenerationConfigError,
    InvalidRequestError,
    ModelInferenceError,
    SynthesisInputTooLongError,
    UnsupportedFeatureError,
)
from pocketsynth.types import (
    LinguisticToken,
    PronunciationOverride,
    RenderedChunk,
    RenderedSegment,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    WordTiming,
)
from pocketsynth.voice_level import VoiceLevelApplication, VoiceLevelConfig


def test_synthesis_segment_requires_nonempty_id_and_string_text() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SynthesisSegment(id="", text="hello")
    with pytest.raises(TypeError, match="text must be a string"):
        SynthesisSegment(id="line-1", text=12)  # type: ignore[arg-type]


def test_rendered_segment_normalizes_audio_and_collections() -> None:
    chunk = RenderedChunk(
        index=0,
        text="hello",
        model_text="hello ",
        token_ids=[1, 2],  # type: ignore[arg-type]
        audio=np.array([0.1, -0.1], dtype=np.float64),
        sample_rate=24_000,
    )
    result = RenderedSegment(
        id="line-1",
        audio=np.array([0.1, -0.1], dtype=np.float64),
        sample_rate=24_000,
        text="hello",
        language="en",
        token_ids=[1, 2],  # type: ignore[arg-type]
        chunks=[chunk],  # type: ignore[arg-type]
        timing=SynthesisTiming(frontend_ms=1.0),
    )

    assert result.audio.dtype == np.float32
    assert result.token_ids == (1, 2)
    assert result.chunks == (chunk,)
    assert result.duration_seconds == pytest.approx(2 / 24_000)


def test_rendered_audio_must_be_finite_mono_and_have_positive_rate() -> None:
    with pytest.raises(ModelInferenceError, match="finite"):
        RenderedSegment("line", np.array([np.nan]), 24_000, "x", "en", ())
    with pytest.raises(ModelInferenceError, match="one-dimensional"):
        RenderedChunk(0, "x", "x", (), np.zeros((1, 2)), 24_000)
    with pytest.raises(ModelInferenceError, match="positive integer"):
        RenderedSegment("line", np.zeros(1), 0, "x", "en", ())


def test_rendered_chunk_and_segment_reject_empty_audio() -> None:
    with pytest.raises(ModelInferenceError, match="must not be empty"):
        RenderedChunk(0, "", "", (), np.zeros(0), 24_000)
    with pytest.raises(ModelInferenceError, match="must not be empty"):
        RenderedSegment("empty", np.zeros(0), 24_000, "", "en", ())


def test_generation_config_contains_only_pocket_inference_controls() -> None:
    config = GenerationConfig(temperature=1.2, lsd_steps=2, max_frames=20, frames_after_eos=0)

    assert (config.temperature, config.lsd_steps, config.max_frames, config.frames_after_eos) == (
        1.2,
        2,
        20,
        0,
    )
    assert not hasattr(config, "normalize_audio")
    assert not hasattr(config, "volume")
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(temperature=2.1)
    with pytest.raises(ValueError, match="lsd_steps"):
        GenerationConfig(lsd_steps=0)
    with pytest.raises(ValueError, match="max_frames"):
        GenerationConfig(max_frames=0)
    with pytest.raises(ValueError, match="frames_after_eos"):
        GenerationConfig(frames_after_eos=-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": float("nan")},
        {"temperature": True},
        {"lsd_steps": 1.5},
        {"max_frames": True},
        {"frames_after_eos": 1.5},
    ],
)
def test_generation_config_rejects_invalid_numeric_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(**kwargs)  # type: ignore[arg-type]


def test_synthesis_request_validates_and_normalizes_linguistic_context() -> None:
    token = LinguisticToken(start=0, end=5, text="hello", pos="NOUN")
    override = PronunciationOverride(start=6, end=11, phonemes="wɜːld")
    request = SynthesisRequest(
        id="seg-1",
        text="hello world",
        language="en",
        tokens=[token],  # type: ignore[arg-type]
        pronunciation_overrides=[override],  # type: ignore[arg-type]
    )

    assert request.tokens == (token,)
    assert request.pronunciation_overrides == (override,)
    with pytest.raises(InvalidRequestError, match="match its source span"):
        SynthesisRequest(
            id="bad-token",
            text="hello",
            tokens=(LinguisticToken(start=0, end=4, text="hello"),),
        )
    with pytest.raises(InvalidRequestError, match="span"):
        SynthesisRequest(
            id="bad-span",
            text="hello",
            pronunciation_overrides=(PronunciationOverride(0, 6, phonemes="həˈloʊ"),),
        )


def test_strict_synthesis_result_normalizes_audio_and_has_no_chunks() -> None:
    timing = WordTiming(
        text="hello",
        char_start=0,
        char_end=5,
        start_sample=0,
        end_sample=100,
        segment_id="seg-1",
    )
    result = SynthesisResult(
        id="seg-1",
        audio=np.array([0.2, -0.2], dtype=np.float64),
        sample_rate=24_000,
        text="hello",
        language="en",
        warnings=["example"],  # type: ignore[arg-type]
        word_timings=[timing],  # type: ignore[arg-type]
        metadata={"token_count": 2},
    )

    assert result.audio.dtype == np.float32
    assert result.warnings == ("example",)
    assert result.word_timings == (timing,)
    assert result.metadata == {"token_count": 2}
    assert result.duration_seconds == pytest.approx(2 / 24_000)
    assert not hasattr(result, "chunks")
    with pytest.raises(ModelInferenceError, match="must not be empty"):
        SynthesisResult("empty", np.zeros(0), 24_000, "hello", "en")


def test_voice_level_config_and_application_types() -> None:
    assert VoiceLevelConfig() == VoiceLevelConfig(mode="off")
    assert VoiceLevelConfig(mode="calibrated", gain_db=-2.5).gain_db == -2.5
    application = VoiceLevelApplication(
        mode="calibrated", gain_db=-2.5, source="catalog", applied=True
    )
    assert application.source == "catalog"
    with pytest.raises(InvalidGenerationConfigError, match="mode"):
        VoiceLevelConfig(mode="dynamic")  # type: ignore[arg-type]
    with pytest.raises(InvalidGenerationConfigError, match="gain_db"):
        VoiceLevelConfig(gain_db=float("inf"))


def test_synthesis_input_limit_and_unsupported_errors_expose_context() -> None:
    too_long = SynthesisInputTooLongError(
        text_length=12, token_count=8, max_tokens=7, bundle_id="english"
    )
    assert isinstance(too_long, ValueError)
    assert (too_long.text_length, too_long.token_count, too_long.max_tokens) == (12, 8, 7)
    assert too_long.bundle_id == "english"
    unsupported = UnsupportedFeatureError(feature="linguistic_tokens")
    assert unsupported.feature == "linguistic_tokens"
    assert "linguistic_tokens" in str(unsupported)

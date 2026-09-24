import numpy as np
import pytest

from pocketsynth.config import GenerationConfig
from pocketsynth.diagnostics import SynthesisTiming
from pocketsynth.errors import ModelInferenceError
from pocketsynth.types import RenderedChunk, RenderedSegment, SynthesisSegment


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

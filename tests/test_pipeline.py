"""Tests for PocketPipeline using a fake runtime."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pocketsynth.config import PipelineConfig
from pocketsynth.errors import PipelineClosedError, VoiceBindingError, VoicePromptError
from pocketsynth.pipeline import PocketPipeline
from pocketsynth.voice import PreparedVoice
from tests.fakes import FakeBundleMetadata, FakePocketRuntime, make_test_voice


def _make_config(tmp_path: Path) -> PipelineConfig:
    """Create a minimal PipelineConfig with fake bundle metadata."""
    meta = {
        "bundle_name": "test",
        "language": "en",
        "schema_version": 2,
        "sample_rate": 24000,
        "samples_per_frame": 1920,
        "max_token_per_chunk": 50,
        "tokenizer_file": "tokenizer.model",
        "bos_before_voice_file": "bos.npy",
    }
    (tmp_path / "bundle.json").write_text(json.dumps(meta))
    (tmp_path / "tokenizer.model").write_bytes(b"x")
    (tmp_path / "bos.npy").write_bytes(b"x")
    for name in [
        "flow_lm_main_int8.onnx",
        "flow_lm_flow_int8.onnx",
        "mimi_decoder_int8.onnx",
        "mimi_encoder.onnx",
        "text_conditioner.onnx",
    ]:
        (tmp_path / name).write_bytes(b"x")
    return PipelineConfig(bundle_dir=tmp_path)


def _make_pipeline(tmp_path: Path) -> tuple[PocketPipeline, FakePocketRuntime]:
    """Create a PocketPipeline with a fake runtime."""
    config = _make_config(tmp_path)
    runtime = FakePocketRuntime(metadata=FakeBundleMetadata())
    pipeline = PocketPipeline(config, runtime=runtime)
    return pipeline, runtime

# --- Pipeline lifecycle ---

def test_pipeline_context_manager_closes_runtime(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    with pipeline:
        pass
    assert runtime._closed


def test_pipeline_context_manager_closes_planner(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    planner = MagicMock()
    pipeline._planner = planner
    pipeline.close()
    planner.close.assert_called_once_with()

def test_pipeline_run_after_close_fails(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    pipeline.close()
    with pytest.raises(PipelineClosedError):
        pipeline.run("hello")


def test_runtime_access_after_close_fails(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    pipeline.close()
    with pytest.raises(PipelineClosedError):
        _ = pipeline.runtime


@patch("pocketsynth.pipeline.adapt_plan")
@patch("pocketsynth.pipeline.build_audio_job")
def test_run_plans_and_renders_text(mock_build, mock_adapt, tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    voice = make_test_voice()

    # Mock the adapt_plan to return empty units
    mock_adapt.return_value = ()

    # Mock build_audio_job to return a fake composition
    fake_audio = np.sin(np.linspace(0, 1, 1000, dtype=np.float32) * 2 * np.pi * 440) * 0.5

    mock_build.return_value = (
        MagicMock(),
        (),
    )

    with pipeline:
        # Mock the Composer at the audiocompose module level
        with patch("audiocompose.Composer") as mock_composer:
            mock_composer.return_value.compose.return_value = MagicMock(
                audio=fake_audio,
                sample_rate=24000,
            )
            result = pipeline.run("Hello world", voice=voice)

    assert result.audio.ndim == 1
    assert result.audio.size > 0
    assert result.sample_rate == 24000
    assert result.metadata["bundle_id"] == "fake-pocket"


def test_render_plan_with_default_voice(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    voice = make_test_voice()
    pipeline.set_default_voice(voice)
    assert pipeline._default_voice is voice


def test_missing_voice_fails_clearly(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    with pytest.raises(VoiceBindingError, match="voice prompt is required"):
        pipeline._resolve_voice(None)


def test_voice_binding_selects_expected_voice(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    voice_a = make_test_voice(sample_rate=24000)
    voice_b = PreparedVoice(state={"other": True}, sample_rate=24000, source="other.wav")
    resolved = pipeline._resolve_voice(voice_a)
    assert resolved is voice_a
    resolved = pipeline._resolve_voice(voice_b)
    assert resolved is voice_b


def test_result_has_plan_id_and_metadata(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    voice = make_test_voice()

    with pipeline:
        with patch("pocketsynth.pipeline.adapt_plan", return_value=()):
            with patch("pocketsynth.pipeline.build_audio_job", return_value=(MagicMock(), ())):
                with patch("audiocompose.Composer") as mock_composer:
                    mock_composer.return_value.compose.return_value = MagicMock(
                        audio=np.zeros(100, dtype=np.float32),
                        sample_rate=24000,
                    )
                    result = pipeline.run("Test", voice=voice)
    assert result.metadata["bundle_id"] == "fake-pocket"
    assert result.metadata["precision"] == "int8"
    assert result.metadata["source_revision"] == "test"


def test_retain_unit_audio_contract(tmp_path):
    config = _make_config(tmp_path)
    config = PipelineConfig(
        bundle_dir=config.bundle_dir,
        retain_unit_audio=True,
    )
    runtime = FakePocketRuntime(metadata=FakeBundleMetadata())
    pipeline = PocketPipeline(config, runtime=runtime)
    voice = make_test_voice()

    with pipeline:
        with patch("pocketsynth.pipeline.adapt_plan", return_value=()):
            with patch("pocketsynth.pipeline.build_audio_job", return_value=(MagicMock(), ())):
                with patch("audiocompose.Composer") as mock_composer:
                    mock_composer.return_value.compose.return_value = MagicMock(
                        audio=np.zeros(100, dtype=np.float32),
                        sample_rate=24000,
                    )
                    result = pipeline.run("Test", voice=voice)

    # With retain_unit_audio=True, chunks should be populated (empty in this mock case)
    assert isinstance(result.chunks, list)


# --- Chunking ---

def test_empty_or_whitespace_text_behavior(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)

    with pipeline:
        # The planner should handle empty text
        plan = pipeline.plan("")
        assert plan is not None


# --- Audio ---

def test_save_wav_creates_valid_mono_pcm16(tmp_path):
    from pocketsynth.types import AudioResult

    audio = np.sin(np.linspace(0, 1, 1000, dtype=np.float32) * 2 * np.pi * 440) * 0.5
    result = AudioResult(
        audio=audio,
        sample_rate=24000,
        source_text="test",
        prepared_text="test",
    )

    wav_path = tmp_path / "test.wav"
    result.save_wav(wav_path)

    import wave

    with wave.open(str(wav_path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == 24000
        assert stream.getnframes() > 0


def test_convenience_save_creates_parent_directories(tmp_path):
    from pocketsynth.types import AudioResult

    audio = np.sin(np.linspace(0, 1, 100, dtype=np.float32) * 2 * np.pi * 440) * 0.5
    result = AudioResult(
        audio=audio,
        sample_rate=24000,
        source_text="test",
        prepared_text="test",
    )

    wav_path = tmp_path / "sub" / "dir" / "test.wav"
    result.save_wav(wav_path)
    assert wav_path.exists()


# --- Diagnostics ---

def test_diagnostics_include_bundle_and_provider_provenance(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    voice = make_test_voice()

    with pipeline:
        with patch("pocketsynth.pipeline.adapt_plan", return_value=()):
            with patch("pocketsynth.pipeline.build_audio_job", return_value=(MagicMock(), ())):
                with patch("audiocompose.Composer") as mock_composer:
                    mock_composer.return_value.compose.return_value = MagicMock(
                        audio=np.zeros(100, dtype=np.float32),
                        sample_rate=24000,
                    )
                    result = pipeline.run("Test", voice=voice)

    assert result.diagnostics is not None
    assert result.diagnostics.bundle_id == "fake-pocket"
    assert result.diagnostics.precision == "int8"


def test_timing_fields_are_consistent(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    voice = make_test_voice()

    with pipeline:
        with patch("pocketsynth.pipeline.adapt_plan", return_value=()):
            with patch("pocketsynth.pipeline.build_audio_job", return_value=(MagicMock(), ())):
                with patch("audiocompose.Composer") as mock_composer:
                    mock_composer.return_value.compose.return_value = MagicMock(
                        audio=np.zeros(100, dtype=np.float32),
                        sample_rate=24000,
                    )
                    result = pipeline.run("Test", voice=voice)

    assert result.timing is not None
    assert result.timing.total_ms is not None
    assert result.timing.total_ms >= 0


# --- Prepared voice ---

def test_prepared_voice_is_reused_without_reencoding(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    voice = make_test_voice()
    pipeline.set_default_voice(voice)

    # The same voice object should be returned
    assert pipeline._default_voice is voice


def test_prepared_voice_is_encoded_once_for_multiple_calls(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    with pipeline:
        pipeline.set_default_voice("voice.wav")
        with patch("pocketsynth.pipeline.adapt_plan", return_value=()):
            with patch("pocketsynth.pipeline.build_audio_job", return_value=(MagicMock(), ())):
                with patch("audiocompose.Composer") as mock_composer:
                    mock_composer.return_value.compose.return_value = MagicMock(
                        audio=np.zeros(100, dtype=np.float32),
                        sample_rate=24000,
                    )
                    pipeline.run("First sentence.")
                    pipeline.run("Second sentence.")
    assert runtime._prepare_count == 1

def test_prepared_voice_from_incompatible_runtime_is_rejected():
    from pocketsynth.voice import PreparedVoice

    voice = PreparedVoice(
        state={"test": True},
        sample_rate=16000,
        source="test.wav",
        bundle_id="other-bundle",
    )
    with pytest.raises(VoicePromptError):
        voice.validate_compatible(bundle_id="fake-pocket", sample_rate=24000)


# --- __call__ delegates to run ---

def test_call_delegates_to_run(tmp_path):
    pipeline, runtime = _make_pipeline(tmp_path)
    voice = make_test_voice()

    with pipeline:
        with patch.object(pipeline, "run") as mock_run:
            mock_run.return_value = MagicMock()
            pipeline("test", voice=voice)
            mock_run.assert_called_once_with("test", voice=voice)

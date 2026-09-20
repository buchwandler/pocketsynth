"""Tests for convenience API functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pocketsynth.convenience import _save_wav_atomically, synthesize, synthesize_to_wav
from pocketsynth.types import AudioResult


def _make_result() -> AudioResult:
    """Create a test AudioResult."""
    audio = np.sin(np.linspace(0, 1, 1000, dtype=np.float32) * 2 * np.pi * 440) * 0.5
    return AudioResult(
        audio=audio,
        sample_rate=24000,
        source_text="test",
        prepared_text="test",
    )


# --- _save_wav_atomically ---

def test_save_wav_atomically_creates_parent_directories(tmp_path):
    result = _make_result()
    dest = tmp_path / "sub" / "dir" / "test.wav"
    _save_wav_atomically(result, dest)
    assert dest.exists()


def test_save_wav_atomically_raises_on_directory(tmp_path):
    result = _make_result()
    with pytest.raises(ValueError, match="is a directory"):
        _save_wav_atomically(result, tmp_path)


def test_save_wav_atomically_is_atomic(tmp_path):
    """Verify the file is written atomically (temp file is cleaned up)."""
    result = _make_result()
    dest = tmp_path / "test.wav"
    _save_wav_atomically(result, dest)
    # No leftover temp files
    assert list(tmp_path.glob(".*.tmp")) == []


# --- synthesize_to_wav ---

def test_synthesize_to_wav_validates_text():
    with pytest.raises(TypeError, match="text must be a string"):
        synthesize_to_wav(123, "out.wav", bundle="test", voice="test.wav")


def test_synthesize_to_wav_validates_bundle():
    with pytest.raises(ValueError, match="bundle must be a non-empty string"):
        synthesize_to_wav("hello", "out.wav", bundle="", voice="test.wav")


@patch("pocketsynth.convenience.PocketPipeline")
def test_synthesize_to_wav_wires_pipeline(mock_pipeline_cls, tmp_path):
    """Verify synthesize_to_wav calls the pipeline correctly."""
    mock_pipeline = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value.__enter__ = MagicMock(return_value=mock_pipeline)
    mock_pipeline_cls.from_pretrained.return_value.__exit__ = MagicMock(return_value=False)

    mock_result = _make_result()
    mock_pipeline.run.return_value = mock_result

    output = tmp_path / "test.wav"
    result = synthesize_to_wav(
        "Hello",
        output,
        bundle="test-bundle",
        voice="test.wav",
    )

    assert result == output
    mock_pipeline.set_default_voice.assert_called_once_with("test.wav")
    mock_pipeline.run.assert_called_once_with("Hello")


# --- synthesize ---

def test_synthesize_validates_text():
    with pytest.raises(TypeError, match="text must be a string"):
        synthesize(123, bundle="test", voice="test.wav")


def test_synthesize_validates_bundle():
    with pytest.raises(ValueError, match="bundle must be a non-empty string"):
        synthesize("hello", bundle="", voice="test.wav")


@patch("pocketsynth.convenience.PocketPipeline")
def test_synthesize_returns_audio_result(mock_pipeline_cls):
    """Verify synthesize returns an AudioResult."""
    mock_pipeline = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value.__enter__ = MagicMock(return_value=mock_pipeline)
    mock_pipeline_cls.from_pretrained.return_value.__exit__ = MagicMock(return_value=False)

    expected = _make_result()
    mock_pipeline.run.return_value = expected

    result = synthesize("Hello", bundle="test-bundle", voice="test.wav")
    assert result is expected

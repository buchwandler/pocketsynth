"""Tests for convenience API functions."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pocketsynth.convenience import _save_wav_atomically, synthesize, synthesize_to_wav
from pocketsynth.types import RenderedSegment
from pocketsynth.voice import PreparedVoice


def _make_result() -> RenderedSegment:
    audio = np.sin(np.linspace(0, 1, 1000, dtype=np.float32) * 2 * np.pi * 440) * 0.5
    return RenderedSegment(
        id="speech",
        audio=audio,
        sample_rate=24_000,
        text="test",
        language="en",
        token_ids=(1, 2, 3),
    )


def _runtime_context(runtime: MagicMock) -> MagicMock:
    context = MagicMock()
    context.__enter__.return_value = runtime
    context.__exit__.return_value = False
    return context


def test_save_wav_atomically_creates_parent_directories_and_valid_wav(tmp_path: Path) -> None:
    destination = tmp_path / "sub" / "dir" / "test.wav"
    _save_wav_atomically(_make_result(), destination)

    with wave.open(str(destination), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == 24_000
        assert stream.getnframes() > 0
    assert list(destination.parent.glob(".*.tmp")) == []


def test_save_wav_atomically_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is a directory"):
        _save_wav_atomically(_make_result(), tmp_path)


def test_synthesize_validates_text_and_bundle() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        synthesize(123, bundle="test", voice="alba")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bundle must be a non-empty string"):
        synthesize("hello", bundle="", voice="alba")


def test_synthesize_uses_runtime_and_prepares_voice() -> None:
    runtime = MagicMock()
    prepared = PreparedVoice(state=object(), sample_rate=24_000, bundle_id="test-bundle")
    runtime.prepare_voice.return_value = prepared
    expected = _make_result()
    runtime.synthesize_text.return_value = expected
    progress = MagicMock()
    context = _runtime_context(runtime)

    with patch(
        "pocketsynth.convenience.PocketRuntime.from_pretrained", return_value=context
    ) as factory:
        result = synthesize(
            "Hello",
            bundle="test-bundle",
            voice="alba",
            sentence_split="none",
            language="en",
            temperature=0.5,
            lsd_steps=2,
            max_frames=100,
            frames_after_eos=0,
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

    assert result is expected
    factory.assert_called_once_with(
        "test-bundle",
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
    runtime.prepare_voice.assert_called_once_with("alba")
    runtime.synthesize_text.assert_called_once()
    args, kwargs = runtime.synthesize_text.call_args
    assert args == ("Hello",)
    assert kwargs["voice"] is prepared
    assert kwargs["language"] == "en"
    assert kwargs["sentence_split"] == "none"
    generation = kwargs["generation"]
    assert (generation.temperature, generation.lsd_steps) == (0.5, 2)
    assert generation.max_frames == 100
    assert generation.frames_after_eos == 0


def test_synthesize_defaults_to_phrasplit() -> None:
    runtime = MagicMock()
    context = _runtime_context(runtime)
    with patch(
        "pocketsynth.convenience.PocketRuntime.from_pretrained",
        return_value=context,
    ):
        synthesize("Hello", bundle="test-bundle", voice="alba")

    _, kwargs = runtime.synthesize_text.call_args
    assert kwargs["sentence_split"] == "phrasplit"


@pytest.mark.parametrize("sentence_split", ["phrasplit", "none"])
def test_synthesize_to_wav_delegates_to_synthesize_and_writes_atomically(
    tmp_path: Path, sentence_split: str
) -> None:
    destination = tmp_path / "nested" / "speech.wav"
    result = _make_result()

    with patch("pocketsynth.convenience.synthesize", return_value=result) as render:
        output = synthesize_to_wav(
            "Hello",
            destination,
            bundle="test-bundle",
            voice="alba",
            language="en",
            sentence_split=sentence_split,
        )

    assert output == destination
    render.assert_called_once_with(
        "Hello",
        bundle="test-bundle",
        voice="alba",
        precision="int8",
        language="en",
        sentence_split=sentence_split,
        temperature=0.7,
        lsd_steps=1,
        max_frames=None,
        frames_after_eos=None,
        providers=None,
        provider_options=None,
        session_options=None,
        cache_dir=None,
        offline=None,
        refresh_catalog=False,
        force_download=False,
        progress=None,
    )
    assert destination.is_file()

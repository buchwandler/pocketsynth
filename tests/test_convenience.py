"""Tests for convenience API functions."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pocketsynth.convenience import (
    _save_wav_atomically,
    synthesize,
    synthesize_to_wav,
    synthesize_with_runtime,
)
from pocketsynth.types import RenderedChunk, RenderedSegment
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


def _make_chunk(text: str = "Hello") -> RenderedChunk:
    return RenderedChunk(
        index=0,
        text=text,
        model_text=text,
        token_ids=(1, 2, 3),
        audio=_make_result().audio,
        sample_rate=24_000,
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
    runtime.bundle_language = "en"
    runtime.sample_rate = 24_000
    runtime.iter_chunks.return_value = [_make_chunk()]
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

    assert result.text == "Hello"
    np.testing.assert_array_equal(result.audio, expected.audio)
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
    runtime.iter_chunks.assert_called_once()
    segment = runtime.iter_chunks.call_args.args[0]
    kwargs = runtime.iter_chunks.call_args.kwargs
    assert segment.text == "Hello"
    assert segment.language == "en"
    assert kwargs["voice"] is prepared
    assert kwargs["generation"].temperature == 0.5
    assert kwargs["generation"].lsd_steps == 2
    generation = kwargs["generation"]
    assert (generation.temperature, generation.lsd_steps) == (0.5, 2)
    assert generation.max_frames == 100
    assert generation.frames_after_eos == 0


def test_synthesize_defaults_to_no_sentence_splitting() -> None:
    runtime = MagicMock()
    runtime.bundle_language = "en"
    runtime.sample_rate = 24_000
    runtime.iter_chunks.return_value = [_make_chunk()]
    context = _runtime_context(runtime)
    with patch(
        "pocketsynth.convenience.PocketRuntime.from_pretrained",
        return_value=context,
    ):
        result = synthesize("Hello", bundle="test-bundle", voice="alba")

    assert result.metadata["sentence_split"] == "none"
    assert runtime.iter_chunks.call_args.args[0].text == "Hello"


def test_synthesize_with_runtime_applies_opt_in_sentence_splitting() -> None:
    runtime = MagicMock()
    runtime.bundle_language = "en"
    runtime.sample_rate = 24_000
    prepared = PreparedVoice(state=object(), sample_rate=24_000, bundle_id="test-bundle")
    runtime.prepare_voice.return_value = prepared
    runtime.iter_chunks.side_effect = [[_make_chunk("First.")], [_make_chunk("Second.")]]
    sentences = ("First.", "Second.")
    with patch("pocketsynth.convenience.split_text_for_synthesis", return_value=sentences) as split:
        result = synthesize_with_runtime(
            runtime, "First. Second.", voice="alba", sentence_split="phrasplit"
        )

    split.assert_called_once_with("First. Second.", language="en", mode="phrasplit")
    assert runtime.iter_chunks.call_count == 2
    assert [call.args[0].text for call in runtime.iter_chunks.call_args_list] == list(sentences)
    assert [chunk.index for chunk in result.chunks] == [0, 1]
    assert result.metadata["sentence_split"] == "phrasplit"
    assert result.text == "First. Second."


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

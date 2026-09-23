import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pocketsynth.errors import VoicePromptError
from pocketsynth.voice import (
    PreparedVoice,
    _read_pcm_wav,
    _resample_linear,
    prepare_voice,
)


def test_read_wav_and_resample(tmp_path):
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.array([0, 1000, -1000, 0], dtype="<i2").tobytes())
    audio, rate = _read_pcm_wav(path)
    assert rate == 16000
    assert audio.dtype == np.float32
    resampled = _resample_linear(audio, 16000, 24000)
    assert resampled.ndim == 1
    assert len(resampled) == 6


@pytest.mark.parametrize("channels,width", [(2, 2), (1, 1)])
def test_read_wav_reports_actual_format(tmp_path: Path, channels: int, width: int):
    path = tmp_path / "invalid-voice.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(16000)
        handle.writeframes(b"\x00" * channels * width)

    with pytest.raises(VoicePromptError, match="Expected mono PCM WAV voice prompt") as caught:
        _read_pcm_wav(path)
    assert f"channels={channels}" in str(caught.value)
    assert f"sample width={width} bytes" in str(caught.value)
    assert "sample rate=16000 Hz" in str(caught.value)


def test_predefined_name_uses_runtime_preparation_without_wav_read() -> None:
    runtime = MagicMock()
    state = object()
    runtime.prepare_predefined_voice.return_value = state

    with patch("pocketsynth.voice._read_pcm_wav") as read_wav:
        voice = prepare_voice(
            runtime,
            "alba",
            sample_rate=24000,
            bundle_id="english_2026-04",
            predefined_voices=("alba",),
        )

    read_wav.assert_not_called()
    runtime.prepare_predefined_voice.assert_called_once_with("alba")
    runtime.prepare_voice.assert_not_called()
    assert voice.state is state
    assert voice.bundle_id == "english_2026-04"
    assert voice.runtime_fingerprint == "english_2026-04"
    assert voice.metadata == {"kind": "predefined", "name": "alba"}


def test_unknown_bare_name_reports_bundle_voices_before_wav_read() -> None:
    runtime = MagicMock()
    with (
        patch("pocketsynth.voice._read_pcm_wav") as read_wav,
        pytest.raises(VoicePromptError, match="Unknown predefined.*Available voices: alba"),
    ):
        prepare_voice(
            runtime,
            "unknown",
            sample_rate=24000,
            predefined_voices=("alba",),
        )
    read_wav.assert_not_called()
    runtime.prepare_predefined_voice.assert_not_called()


def test_path_object_matching_voice_name_remains_a_local_prompt() -> None:
    runtime = MagicMock()
    audio = np.ones(4, dtype=np.float32)
    source = Path("alba")
    with patch("pocketsynth.voice._read_pcm_wav", return_value=(audio, 24000)) as read_wav:
        voice = prepare_voice(
            runtime,
            source,
            sample_rate=24000,
            bundle_id="english_2026-04",
            predefined_voices=("alba",),
        )

    read_wav.assert_called_once_with(source)
    runtime.prepare_predefined_voice.assert_not_called()
    runtime.prepare_voice.assert_called_once()
    assert voice.source == str(source)


def test_local_file_string_remains_a_reference_prompt(tmp_path: Path) -> None:
    runtime = MagicMock()
    audio = np.ones(4, dtype=np.float32)
    source = str(tmp_path / "reference.wav")
    with patch("pocketsynth.voice._read_pcm_wav", return_value=(audio, 24000)) as read_wav:
        voice = prepare_voice(
            runtime,
            source,
            sample_rate=24000,
            predefined_voices=("alba",),
        )

    read_wav.assert_called_once_with(source)
    runtime.prepare_predefined_voice.assert_not_called()
    assert voice.source == source


def test_in_memory_audio_and_prepared_voice_paths_are_preserved() -> None:
    runtime = MagicMock()
    audio = np.ones(4, dtype=np.float32)
    prepared = PreparedVoice(state=object(), sample_rate=24000)

    assert (
        prepare_voice(
            runtime,
            (audio, 24000),
            sample_rate=24000,
            predefined_voices=("alba",),
        ).sample_rate
        == 24000
    )
    assert (
        prepare_voice(
            runtime,
            prepared,
            sample_rate=24000,
            predefined_voices=("alba",),
        )
        is prepared
    )
    runtime.prepare_predefined_voice.assert_not_called()
    runtime.prepare_voice.assert_called_once()


def test_invalid_pathlike_is_reported_as_voice_prompt_error() -> None:
    class InvalidPath:
        def __fspath__(self) -> str:
            raise TypeError("invalid path")

    with pytest.raises(VoicePromptError, match="could not read"):
        prepare_voice(
            MagicMock(),
            InvalidPath(),
            sample_rate=24000,
            predefined_voices=("alba",),
        )

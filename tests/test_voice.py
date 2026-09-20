import wave
from pathlib import Path

import numpy as np
import pytest

from pocketsynth.errors import VoicePromptError
from pocketsynth.voice import _read_pcm_wav, _resample_linear


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

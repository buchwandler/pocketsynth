import wave

import numpy as np

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

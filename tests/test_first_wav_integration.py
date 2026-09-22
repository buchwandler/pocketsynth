from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from pocketsynth import PocketPipeline

pytestmark = pytest.mark.integration


def test_configured_local_first_wav_is_valid(tmp_path: Path) -> None:
    bundle_raw = os.environ.get("POCKETSYNTH_TEST_BUNDLE_DIR")
    voice = os.environ.get("POCKETSYNTH_TEST_VOICE_WAV")
    if not bundle_raw or not voice:
        pytest.skip("set POCKETSYNTH_TEST_BUNDLE_DIR and POCKETSYNTH_TEST_VOICE_WAV")

    output = tmp_path / "first-wav.wav"
    with PocketPipeline.load(bundle_raw) as pipeline:
        pipeline.set_default_voice(voice)
        result = pipeline("Hello from the PocketSynth first WAV integration test.")
        result.save_wav(output)

    with wave.open(str(output), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getnframes() > 0
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")

    assert samples.size > 0
    assert np.all(np.isfinite(samples))
    assert np.max(np.abs(samples)) > 0

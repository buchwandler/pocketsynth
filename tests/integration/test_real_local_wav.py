from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from pocketsynth import BundleMetadata, PocketPipeline

pytestmark = pytest.mark.integration


def _assert_wav(path: Path, *, sample_rate: int) -> None:
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == sample_rate
        frames = stream.readframes(stream.getnframes())
        assert stream.getnframes() > 0
    samples = np.frombuffer(frames, dtype="<i2")
    assert samples.size > 0
    assert np.max(np.abs(samples)) > 0


def test_real_local_bundle_produces_wav(tmp_path: Path) -> None:
    bundle_raw = os.environ.get("POCKETSYNTH_TEST_BUNDLE_DIR")
    voice = os.environ.get("POCKETSYNTH_TEST_VOICE_WAV")
    if not bundle_raw or not voice:
        pytest.skip("set POCKETSYNTH_TEST_BUNDLE_DIR and POCKETSYNTH_TEST_VOICE_WAV")

    bundle = Path(bundle_raw)
    expected_rate = BundleMetadata.load(bundle / "bundle.json").sample_rate
    output = tmp_path / "local.wav"
    with PocketPipeline.load(bundle, precision="int8") as pipeline:
        pipeline.set_default_voice(voice)
        result = pipeline("Local Pocket smoke test.")
        assert result.sample_rate == expected_rate
        assert result.duration_seconds > 0
        assert np.all(np.isfinite(result.audio))
        assert float(np.max(np.abs(result.audio))) > 0
        result.save_wav(output)

    _assert_wav(output, sample_rate=expected_rate)

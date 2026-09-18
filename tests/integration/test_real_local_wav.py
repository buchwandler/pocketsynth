"""Integration smoke test for real Pocket bundles.

Requires external model assets. Skipped unless environment variables are set.
"""

from __future__ import annotations

import os
import wave

import numpy as np
import pytest

from pocketsynth import PocketPipeline


@pytest.mark.integration
def test_real_local_bundle_produces_nonempty_wav(tmp_path):
    """Open a real local Pocket bundle, synthesize, and validate the WAV."""
    bundle = os.environ.get("POCKETSYNTH_TEST_BUNDLE_DIR")
    voice = os.environ.get("POCKETSYNTH_TEST_VOICE_WAV")

    if not bundle or not voice:
        pytest.skip("real Pocket assets not configured")

    output = tmp_path / "smoke.wav"

    with PocketPipeline.load(bundle, precision="int8") as pipeline:
        pipeline.set_default_voice(voice)
        result = pipeline.run("Pocket synthesis smoke test.")
        result.save_wav(output)

    assert result.audio.ndim == 1
    assert result.audio.size > 0
    assert np.isfinite(result.audio).all()
    assert float(np.max(np.abs(result.audio))) > 0.0

    with wave.open(str(output), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == result.sample_rate
        assert stream.getnframes() > 0

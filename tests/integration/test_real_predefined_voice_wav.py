from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from pocketsynth import PocketPipeline

pytestmark = [pytest.mark.integration, pytest.mark.network]


def _assert_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == 24_000
        frames = stream.readframes(stream.getnframes())
        assert stream.getnframes() > 0

    samples = np.frombuffer(frames, dtype="<i2")
    assert samples.size > 0
    assert np.max(np.abs(samples)) > 0


def _render(cache_dir: Path, output: Path, *, offline: bool) -> None:
    with PocketPipeline.from_pretrained(
        "english_2026-04",
        precision="int8",
        cache_dir=cache_dir,
        offline=offline,
    ) as pipeline:
        assert "alba" in pipeline.predefined_voices
        pipeline.set_default_voice("alba")
        result = pipeline("Hello from the PocketSynth predefined voice test.")
        assert result.sample_rate == 24_000
        assert result.duration_seconds > 0
        assert np.all(np.isfinite(result.audio))
        assert float(np.max(np.abs(result.audio))) > 0
        result.save_wav(output)

    _assert_wav(output)


def test_managed_alba_produces_online_and_cached_offline_wavs(tmp_path: Path) -> None:
    if os.environ.get("POCKETSYNTH_TEST_PREDEFINED_VOICE") != "1":
        pytest.skip(
            "set POCKETSYNTH_TEST_PREDEFINED_VOICE=1 after configuring gated Hugging Face access"
        )

    cache_dir = tmp_path / "cache"
    online = tmp_path / "alba-online.wav"
    _render(cache_dir, online, offline=False)

    offline = tmp_path / "alba-offline.wav"
    _render(cache_dir, offline, offline=True)

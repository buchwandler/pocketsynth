from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from pocketsynth import PocketPipeline

pytestmark = [pytest.mark.integration, pytest.mark.network]


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


def _render(bundle: str, voice: str, cache_dir: Path, output: Path, *, offline: bool) -> int:
    with PocketPipeline.from_pretrained(
        bundle,
        precision="int8",
        cache_dir=cache_dir,
        offline=offline,
    ) as pipeline:
        pipeline.set_default_voice(voice)
        result = pipeline("Managed Pocket smoke test.")
        assert result.duration_seconds > 0
        assert np.all(np.isfinite(result.audio))
        assert float(np.max(np.abs(result.audio))) > 0
        result.save_wav(output)
        return result.sample_rate


def test_managed_english_bundle_produces_wav(tmp_path: Path) -> None:
    voice = os.environ.get("POCKETSYNTH_TEST_VOICE_WAV")
    if not voice:
        pytest.skip("set POCKETSYNTH_TEST_VOICE_WAV")

    bundle = os.environ.get("POCKETSYNTH_TEST_BUNDLE", "english_2026-04")
    cache_dir = tmp_path / "cache"
    first = tmp_path / "managed.wav"
    sample_rate = _render(bundle, voice, cache_dir, first, offline=False)
    _assert_wav(first, sample_rate=sample_rate)

    cached = tmp_path / "managed-offline.wav"
    cached_rate = _render(bundle, voice, cache_dir, cached, offline=True)
    assert cached_rate == sample_rate
    _assert_wav(cached, sample_rate=sample_rate)

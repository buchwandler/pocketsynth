"""Render already-prepared speakable text from a managed Pocket bundle."""

from __future__ import annotations

import os
from pathlib import Path

from _output import artefact_path

from pocketsynth import PocketRuntime

BUNDLE = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04")
VOICE_RAW = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")
TEXT = "Hello from PocketSynth. This is prepared speakable text."

if not VOICE_RAW:
    raise SystemExit("Set POCKETSYNTH_EXAMPLE_VOICE to a mono 16-bit PCM reference WAV.")

voice_path = Path(VOICE_RAW)
with PocketRuntime.from_pretrained(
    BUNDLE,
    precision="int8",
    offline=os.environ.get("POCKETSYNTH_EXAMPLE_OFFLINE") == "1",
) as runtime:
    voice = runtime.prepare_voice(voice_path)
    result = runtime.synthesize_text(TEXT, voice=voice)
    wav_path = artefact_path("basic.wav")
    result.save_wav(wav_path)

print(f"Bundle: {BUNDLE}")
print(f"Voice: {voice_path}")
print(f"WAV path: {wav_path}")
print(f"Sample rate: {result.sample_rate}")
print(f"Duration: {result.duration_seconds:.3f}s")

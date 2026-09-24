"""Render already-prepared speakable text from a local Pocket bundle."""

from __future__ import annotations

import os
from pathlib import Path

from _output import artefact_path

from pocketsynth import PocketRuntime

TEXT = "Hello from PocketSynth. This is prepared speakable text."
bundle_dir_raw = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE_DIR")
voice_raw = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")

if not bundle_dir_raw:
    raise SystemExit("Set POCKETSYNTH_EXAMPLE_BUNDLE_DIR to a local Pocket ONNX bundle directory.")
if not voice_raw:
    raise SystemExit("Set POCKETSYNTH_EXAMPLE_VOICE to a mono 16-bit PCM reference WAV.")

bundle_dir = Path(bundle_dir_raw)
voice_path = Path(voice_raw)
with PocketRuntime.load(bundle_dir, precision="int8") as runtime:
    voice = runtime.prepare_voice(voice_path)
    result = runtime.synthesize_text(TEXT, voice=voice)
    wav_path = artefact_path("basic_local.wav")
    result.save_wav(wav_path)

print(f"Bundle: {bundle_dir}")
print(f"Voice: {voice_path}")
print(f"WAV path: {wav_path}")
print(f"Sample rate: {result.sample_rate}")
print(f"Duration: {result.duration_seconds:.3f}s")

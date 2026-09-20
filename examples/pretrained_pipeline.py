"""Demonstrate prepared-voice reuse with PocketPipeline."""

from __future__ import annotations

import os
from pathlib import Path

from _output import artefact_path

from pocketsynth import PocketPipeline

BUNDLE = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04")
VOICE_RAW = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")

if not VOICE_RAW:
    raise SystemExit("Set POCKETSYNTH_EXAMPLE_VOICE to a mono 16-bit PCM reference WAV.")

voice_path = Path(VOICE_RAW)

with PocketPipeline.from_pretrained(
    BUNDLE,
    precision="int8",
    offline=os.environ.get("POCKETSYNTH_EXAMPLE_OFFLINE") == "1",
) as pipeline:
    narrator = pipeline.prepare_voice(voice_path)
    pipeline.set_default_voice(narrator)

    first = pipeline.run("This is the first sentence.")
    second = pipeline.run("This is the second sentence.")

    first_path = artefact_path("pretrained_01.wav")
    second_path = artefact_path("pretrained_02.wav")

    first.save_wav(first_path)
    second.save_wav(second_path)

    print(f"Bundle: {BUNDLE}")
    print(f"Voice: {voice_path}")
    print(f"First WAV: {first_path} ({first.duration_seconds:.3f}s)")
    print(f"Second WAV: {second_path} ({second.duration_seconds:.3f}s)")
    print("Voice was prepared once and reused for both sentences.")

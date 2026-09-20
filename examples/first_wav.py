"""Smallest managed Pocket bundle -> WAV example."""

from __future__ import annotations

import os

from _output import artefact_path

from pocketsynth import synthesize_to_wav

voice = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")
if not voice:
    raise SystemExit("Set POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference.wav")

output = artefact_path("first_wav.wav")
synthesize_to_wav(
    "Hello from PocketSynth.",
    output,
    bundle=os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04"),
    voice=voice,
    precision="int8",
    offline=os.environ.get("POCKETSYNTH_EXAMPLE_OFFLINE") == "1",
)
print(f"Wrote {output.resolve()}")

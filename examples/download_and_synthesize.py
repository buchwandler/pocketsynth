"""Managed Pocket bundle download with progress and first audio output."""

from __future__ import annotations

import os

from _output import artefact_path

from pocketsynth import ConsoleAssetProgress, synthesize_to_wav

voice = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")
if not voice:
    raise SystemExit("Set POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference.wav")

output = artefact_path("download_and_synthesize.wav")
synthesize_to_wav(
    "Hello from a managed Pocket bundle.",
    output,
    bundle=os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04"),
    voice=voice,
    precision="int8",
    progress=ConsoleAssetProgress(),
)
print(f"Wrote {output.resolve()}")

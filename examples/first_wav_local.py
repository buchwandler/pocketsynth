"""Smallest local Pocket bundle-to-WAV example."""

from __future__ import annotations

import os
from pathlib import Path

from _output import artefact_path

from pocketsynth import PocketRuntime

bundle_raw = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE_DIR")
voice = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")
if not bundle_raw:
    raise SystemExit("Set POCKETSYNTH_EXAMPLE_BUNDLE_DIR=/path/to/pocket-bundle")
if not voice:
    raise SystemExit("Set POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference.wav")

output = artefact_path("first_wav_local.wav")
with PocketRuntime.load(Path(bundle_raw), precision="int8") as runtime:
    runtime.synthesize_text("Hello from Pocket.", voice=voice).save_wav(output)

print(f"Wrote {output.resolve()}")

"""Smallest local Pocket bundle -> UtterPlan -> WAV example."""

from __future__ import annotations

import os
from pathlib import Path

from _output import artefact_path

from pocketsynth import PocketPipeline

TEXT = "Hello from PocketSynth. This audio was rendered from a saved UtterancePlan."

bundle_dir_raw = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE_DIR")
voice_raw = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")

if not bundle_dir_raw:
    raise SystemExit(
        "Set POCKETSYNTH_EXAMPLE_BUNDLE_DIR to a local Pocket ONNX bundle directory."
    )

if not voice_raw:
    raise SystemExit(
        "Set POCKETSYNTH_EXAMPLE_VOICE to a mono 16-bit PCM reference WAV."
    )

bundle_dir = Path(bundle_dir_raw)
voice_path = Path(voice_raw)

with PocketPipeline.load(
    bundle_dir,
    precision="int8",
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    pipeline.set_default_voice(voice_path)

    plan = pipeline.plan(TEXT, unit="sentence")
    plan_path = artefact_path("basic_local.utterplan.json")
    plan.save(plan_path)

    result = pipeline.render_plan(plan)
    wav_path = artefact_path("basic_local.wav")
    result.save_wav(wav_path)

    print(f"Bundle: {bundle_dir}")
    print(f"Voice: {voice_path}")
    print(f"Plan: {plan.plan_id}")
    print(f"Spoken text: {plan.texts.spoken}")
    print(f"Plan path: {plan_path}")
    print(f"WAV path: {wav_path}")
    print(f"Sample rate: {result.sample_rate}")
    print(f"Duration: {result.duration_seconds:.3f}s")

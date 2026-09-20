"""Smallest managed Pocket bundle -> UtterPlan -> WAV example."""

from __future__ import annotations

import os
from pathlib import Path

from _output import artefact_path

from pocketsynth import PocketPipeline

BUNDLE = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04")
VOICE_RAW = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")
TEXT = "Hello from PocketSynth. This audio was rendered from a saved UtterancePlan."

if not VOICE_RAW:
    raise SystemExit(
        "Set POCKETSYNTH_EXAMPLE_VOICE to a mono 16-bit PCM reference WAV."
    )

voice_path = Path(VOICE_RAW)

with PocketPipeline.from_pretrained(
    BUNDLE,
    precision="int8",
    offline=os.environ.get("POCKETSYNTH_EXAMPLE_OFFLINE") == "1",
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    pipeline.set_default_voice(voice_path)

    plan = pipeline.plan(TEXT, unit="sentence")
    plan_path = artefact_path("basic.utterplan.json")
    plan.save(plan_path)

    result = pipeline.render_plan(plan)
    wav_path = artefact_path("basic.wav")
    result.save_wav(wav_path)

    print(f"Bundle: {BUNDLE}")
    print(f"Voice: {voice_path}")
    print(f"Plan: {plan.plan_id}")
    print(f"Spoken text: {plan.texts.spoken}")
    print(f"Plan path: {plan_path}")
    print(f"WAV path: {wav_path}")
    print(f"Sample rate: {result.sample_rate}")
    print(f"Duration: {result.duration_seconds:.3f}s")

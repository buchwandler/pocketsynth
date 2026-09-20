"""Demonstrate semantic reproducibility: plan -> save -> reload -> render."""

from __future__ import annotations

import os
from pathlib import Path

from _output import artefact_path
from utterplan import UtterancePlan

from pocketsynth import PocketPipeline

BUNDLE = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04")
VOICE_RAW = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")
TEXT = "This sentence was planned, saved to disk, reloaded, and then rendered."

if not VOICE_RAW:
    raise SystemExit("Set POCKETSYNTH_EXAMPLE_VOICE to a mono 16-bit PCM reference WAV.")

voice_path = Path(VOICE_RAW)

with PocketPipeline.from_pretrained(
    BUNDLE,
    precision="int8",
    offline=os.environ.get("POCKETSYNTH_EXAMPLE_OFFLINE") == "1",
) as pipeline:
    pipeline.set_default_voice(voice_path)

    # Plan
    plan = pipeline.plan(TEXT, unit="sentence")
    plan_path = artefact_path("roundtrip.utterplan.json")
    plan.save(plan_path)
    print(f"Original plan: {plan.plan_id}")
    print(f"Plan saved to: {plan_path}")

    # Reload
    reloaded = UtterancePlan.load(plan_path)
    print(f"Reloaded plan: {reloaded.plan_id}")
    assert reloaded.plan_id == plan.plan_id, "Plan ID changed after roundtrip"

    # Render from reloaded plan
    result = pipeline.render_plan(reloaded)
    wav_path = artefact_path("roundtrip.wav")
    result.save_wav(wav_path)

    print(f"WAV path: {wav_path}")
    print(f"Sample rate: {result.sample_rate}")
    print(f"Duration: {result.duration_seconds:.3f}s")
    print("Plan roundtrip succeeded: semantic plan survives save/reload.")

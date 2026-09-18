"""Run all PocketSynth examples and validate outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave
from array import array
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXAMPLES_DIR.parent
_ARTEFACT_DIR = _PROJECT_ROOT / "example-artefacts"

EXAMPLES = [
    "basic_local.py",
    "basic.py",
    "pretrained_pipeline.py",
    "plan_roundtrip.py",
]


def _validate_wav(path: Path) -> None:
    """Validate that a WAV file is mono 16-bit PCM with non-zero audio."""
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1:
            raise RuntimeError(f"{path} must be mono")
        if stream.getsampwidth() != 2:
            raise RuntimeError(f"{path} must be 16-bit PCM")
        if stream.getframerate() <= 0:
            raise RuntimeError(f"{path} has invalid sample rate")
        frames = stream.readframes(stream.getnframes())
        if not frames:
            raise RuntimeError(f"{path} contains no audio frames")

        samples = array("h")
        samples.frombytes(frames)
        if samples and max(abs(sample) for sample in samples) == 0:
            raise RuntimeError(f"{path} contains only digital silence")


def _validate_plan(path: Path) -> None:
    """Validate that an UtterPlan JSON file is loadable."""
    from utterplan import UtterancePlan

    plan = UtterancePlan.load(path)
    plan.validate()


def _run_example(name: str, *, env: dict[str, str]) -> Path:
    """Run a single example script as a subprocess."""
    script = _EXAMPLES_DIR / name
    output_dir = _ARTEFACT_DIR / f"examples__{name.replace('.py', '')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_env = dict(os.environ)
    run_env.update(env)
    run_env["POCKETSYNTH_EXAMPLE_OUTPUT_DIR"] = str(output_dir)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_EXAMPLES_DIR),
        env=run_env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed (exit {result.returncode}):\n{result.stderr}"
        )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all PocketSynth examples")
    parser.add_argument("--list", action="store_true", help="List examples and exit")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    args = parser.parse_args(argv)

    if args.list:
        for name in EXAMPLES:
            print(name)
        return 0

    bundle_dir = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE_DIR")
    bundle = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04")
    voice = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")

    if not voice:
        print("ERROR: Set POCKETSYNTH_EXAMPLE_VOICE to a mono 16-bit PCM reference WAV.")
        return 1

    env: dict[str, str] = {"POCKETSYNTH_EXAMPLE_VOICE": voice}
    if bundle_dir:
        env["POCKETSYNTH_EXAMPLE_BUNDLE_DIR"] = bundle_dir
    env["POCKETSYNTH_EXAMPLE_BUNDLE"] = bundle

    # Filter examples based on available assets
    runnable = []
    for name in EXAMPLES:
        if name == "basic_local.py" and not bundle_dir:
            print(f"SKIP {name} (POCKETSYNTH_EXAMPLE_BUNDLE_DIR not set)")
            continue
        runnable.append(name)

    if not runnable:
        print("No examples to run. Set POCKETSYNTH_EXAMPLE_BUNDLE_DIR for local examples.")
        return 0

    failures: list[str] = []
    for name in runnable:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            output_dir = _run_example(name, env=env)

            # Validate artifacts
            plans = list(output_dir.glob("*.utterplan.json"))
            wavs = list(output_dir.glob("*.wav"))

            if not plans:
                raise RuntimeError(f"No .utterplan.json found in {output_dir}")
            if not wavs:
                raise RuntimeError(f"No .wav found in {output_dir}")

            for plan_path in plans:
                _validate_plan(plan_path)
            for wav_path in wavs:
                _validate_wav(wav_path)

            print(f"OK ({len(plans)} plans, {len(wavs)} WAVs)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            failures.append(name)
            if args.fail_fast:
                break

    if failures:
        print(f"\n{len(failures)} example(s) failed: {', '.join(failures)}")
        return 1

    print(f"\nAll {len(runnable)} example(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

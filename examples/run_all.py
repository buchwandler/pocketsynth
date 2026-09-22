"""Run selected PocketSynth examples without surprising network downloads."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import wave
from array import array
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXAMPLES_DIR.parent
_ARTEFACT_DIR = _PROJECT_ROOT / "example-artefacts"

EXAMPLES = (
    ("quickstart.py", "managed", False),
    ("local_bundle.py", "local", False),
    ("basic.py", "managed", True),
    ("pretrained_pipeline.py", "managed", False),
    ("plan_roundtrip.py", "managed", True),
    ("basic_local.py", "local", True),
    ("download_and_synthesize.py", "managed", False),
    ("first_wav.py", "managed", False),
    ("first_wav_local.py", "local", False),
)


def _validate_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1:
            raise RuntimeError(f"{path} must be mono")
        if stream.getsampwidth() != 2:
            raise RuntimeError(f"{path} must be 16-bit PCM")
        if stream.getframerate() <= 0:
            raise RuntimeError(f"{path} has invalid sample rate")
        frames = stream.readframes(stream.getnframes())
        if stream.getnframes() <= 0:
            raise RuntimeError(f"{path} contains no audio frames")
    samples = array("h")
    samples.frombytes(frames)
    if not samples or max(abs(sample) for sample in samples) == 0:
        raise RuntimeError(f"{path} contains only digital silence")


def _validate_plan(path: Path) -> None:
    from utterplan import UtterancePlan

    plan = UtterancePlan.load(path)
    plan.validate()


def _run_example(name: str, *, env: dict[str, str]) -> Path:
    script = _EXAMPLES_DIR / name
    output_dir = _ARTEFACT_DIR / f"examples__{name.replace('.py', '')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script)]
    if name == "quickstart.py":
        command.extend(
            [
                "--voice",
                env["POCKETSYNTH_EXAMPLE_VOICE"],
                "--output",
                str(output_dir / "quickstart.wav"),
            ]
        )
    elif name == "local_bundle.py":
        command.extend(
            [
                "--bundle-dir",
                env["POCKETSYNTH_EXAMPLE_BUNDLE_DIR"],
                "--voice",
                env["POCKETSYNTH_EXAMPLE_VOICE"],
                "--output",
                str(output_dir / "local_bundle.wav"),
            ]
        )
    run_env = dict(os.environ)
    run_env.update(env)
    run_env["POCKETSYNTH_EXAMPLE_OUTPUT_DIR"] = str(output_dir)
    result = subprocess.run(
        command,
        cwd=str(_EXAMPLES_DIR),
        env=run_env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed (exit {result.returncode}):\n{result.stderr}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PocketSynth examples")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--managed-only", action="store_true", help="Run managed examples only")
    mode.add_argument("--local-only", action="store_true", help="Run local examples only")
    parser.add_argument("--include-network", action="store_true", help="Enable managed downloads")
    parser.add_argument("--offline", action="store_true", help="Use cached managed assets only")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--list", action="store_true", help="List examples and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, kind, needs_plan in EXAMPLES:
            suffix = "\tplan" if needs_plan else ""
            print(f"{name}\t{kind}{suffix}")
        return 0

    bundle_dir = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE_DIR")
    voice = os.environ.get("POCKETSYNTH_EXAMPLE_VOICE")
    include_managed = args.managed_only or args.include_network
    include_local = not args.managed_only
    if args.local_only:
        include_local = True
        include_managed = False

    runnable: list[tuple[str, str, bool]] = []
    for example in EXAMPLES:
        name, kind, needs_plan = example
        if kind == "local":
            if not include_local or not bundle_dir:
                reason = (
                    "local mode disabled"
                    if not include_local
                    else "POCKETSYNTH_EXAMPLE_BUNDLE_DIR not set"
                )
                print(f"SKIP {name} ({reason})")
                continue
        elif not include_managed:
            print(f"SKIP {name} (managed downloads disabled; use --include-network)")
            continue
        runnable.append(example)

    if runnable and not voice:
        print("ERROR: Set POCKETSYNTH_EXAMPLE_VOICE to a mono 16-bit PCM reference WAV.")
        return 1
    if not runnable:
        print("No examples to run. Configure a local bundle or pass --include-network.")
        return 0

    env = {"POCKETSYNTH_EXAMPLE_VOICE": voice or ""}
    if bundle_dir:
        env["POCKETSYNTH_EXAMPLE_BUNDLE_DIR"] = bundle_dir
    env["POCKETSYNTH_EXAMPLE_BUNDLE"] = os.environ.get(
        "POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04"
    )
    if args.offline:
        env["POCKETSYNTH_EXAMPLE_OFFLINE"] = "1"

    failures: list[str] = []
    for name, _kind, needs_plan in runnable:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            output_dir = _run_example(name, env=env)
            plans = list(output_dir.glob("*.utterplan.json"))
            wavs = list(output_dir.glob("*.wav"))
            if needs_plan and not plans:
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

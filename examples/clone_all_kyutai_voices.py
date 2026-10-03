"""Synthesize one sample for every cataloged Kyutai WAV prompt."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    from _output import artifact_path
except ModuleNotFoundError:  # imported as examples.clone_all_kyutai_voices
    from examples._output import artifact_path

from pocketsynth import PocketRuntime, list_voice_prompts

DEFAULT_BUNDLE = "english_2026-04"
DEFAULT_TEXT = "Hello. This is a PocketSynth voice-cloning sample."
MANIFEST_FIELDS = ("ref", "source_path", "license", "variant", "output", "status", "error")


def safe_output_name(prompt_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", prompt_id) + ".wav"


def _is_noncommercial(license_name: str) -> bool:
    tokens = re.sub(r"[^a-z0-9]+", "-", license_name.casefold()).strip("-").split("-")
    return (
        "nc" in tokens
        or "noncommercial" in tokens
        or {
            "non",
            "commercial",
        }.issubset(tokens)
    )


def _nonnegative_int(value: str) -> int:
    limit = int(value)
    if limit < 0:
        raise argparse.ArgumentTypeError("limit must be zero or greater")
    return limit


def _write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> None:
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: example-artifacts/kyutai-voices)",
    )
    parser.add_argument("--dataset")
    parser.add_argument("--variant")
    parser.add_argument("--limit", type=_nonnegative_int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--exclude-noncommercial",
        action="store_true",
        help="filter catalog license labels mechanically; this is not legal advice",
    )
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir or artifact_path("kyutai-voices")

    prompts = list_voice_prompts(
        cache_dir=args.cache_dir,
        offline=args.offline,
        dataset=args.dataset,
        variant=args.variant,
    )
    if args.exclude_noncommercial:
        prompts = tuple(prompt for prompt in prompts if not _is_noncommercial(prompt.license))
    if args.limit is not None:
        prompts = prompts[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if args.dry_run:
        for prompt in prompts:
            destination = args.output_dir / safe_output_name(
                prompt.ref.removeprefix("kyutai-tts-voices:")
            )
            rows.append(
                {
                    "ref": prompt.ref,
                    "source_path": prompt.source_path,
                    "license": prompt.license,
                    "variant": prompt.variant,
                    "output": str(destination),
                    "status": "dry-run",
                    "error": "",
                }
            )
            print(f"DRY RUN {prompt.ref} -> {destination}")
    elif prompts:
        with PocketRuntime.from_pretrained(
            args.bundle, cache_dir=args.cache_dir, offline=args.offline
        ) as runtime:
            for index, prompt in enumerate(prompts, start=1):
                destination = args.output_dir / safe_output_name(
                    prompt.ref.removeprefix("kyutai-tts-voices:")
                )
                row = {
                    "ref": prompt.ref,
                    "source_path": prompt.source_path,
                    "license": prompt.license,
                    "variant": prompt.variant,
                    "output": str(destination),
                    "status": "ok",
                    "error": "",
                }
                try:
                    voice = runtime.prepare_voice(prompt.ref)
                    audio = runtime.synthesize_text(args.text, voice=voice)
                    audio.save_wav(destination)
                    print(f"[{index}/{len(prompts)}] {prompt.ref} -> {destination}")
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"[{index}/{len(prompts)}] ERROR {prompt.ref}: {exc}")
                rows.append(row)

    _write_manifest(args.output_dir, rows)
    print(f"Manifest: {args.output_dir / 'manifest.csv'}")
    return 1 if any(row["status"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

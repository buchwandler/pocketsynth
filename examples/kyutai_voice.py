"""Synthesize with a cataloged Kyutai reference prompt, without manual downloading."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from _output import artifact_path
except ModuleNotFoundError:  # imported as examples.kyutai_voice
    from examples._output import artifact_path
from pocketsynth import PocketRuntime, inspect_voice_prompt

DEFAULT_BUNDLE = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04")
DEFAULT_REF = "kyutai-tts-voices:alba-mackenna/casual"
TEXT = "Hello, this voice was cloned from a managed Kyutai reference recording."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--voice", default=DEFAULT_REF, help=f"managed reference (default: {DEFAULT_REF})"
    )
    parser.add_argument(
        "--pin-prompt",
        action="store_true",
        help="inspect current catalog metadata and pin its identity before fetching audio",
    )
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--offline",
        action="store_true",
        default=os.environ.get("POCKETSYNTH_EXAMPLE_OFFLINE") == "1",
        help="use cached bundle and reference assets only",
    )
    args = parser.parse_args(argv)
    output = args.output or artifact_path("kyutai_casual.wav")
    voice_source = args.voice
    if args.pin_prompt:
        voice_source = inspect_voice_prompt(
            args.voice, cache_dir=args.cache_dir, offline=args.offline
        )

    with PocketRuntime.from_pretrained(
        args.bundle, cache_dir=args.cache_dir, offline=args.offline
    ) as runtime:
        voice = runtime.prepare_voice(voice_source)
        audio = runtime.synthesize_text(args.text, voice=voice)
        audio.save_wav(output)
    print(f"WAV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

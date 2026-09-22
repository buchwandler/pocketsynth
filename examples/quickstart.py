"""Minimal managed PocketSynth text-to-WAV example."""

from __future__ import annotations

import argparse
from pathlib import Path

from pocketsynth import synthesize_to_wav


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize one PocketSynth WAV")
    parser.add_argument("--voice", type=Path, required=True, help="mono 16-bit PCM reference WAV")
    parser.add_argument("--output", type=Path, required=True, help="output WAV path")
    parser.add_argument("--bundle", default="english_2026-04")
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("text", nargs="?", default="Hello from Pocket.")
    args = parser.parse_args(argv)

    output = synthesize_to_wav(
        args.text,
        args.output,
        bundle=args.bundle,
        voice=args.voice,
        precision=args.precision,
        cache_dir=args.cache_dir,
        offline=args.offline,
        refresh_catalog=args.refresh_catalog,
        force_download=args.force_download,
    )
    print(f"Wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

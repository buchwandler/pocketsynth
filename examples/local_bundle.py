"""Minimal local Pocket bundle text-to-WAV example."""

from __future__ import annotations

import argparse
from pathlib import Path

from pocketsynth import PocketPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize from a local Pocket bundle")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True, help="mono 16-bit PCM reference WAV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("text", nargs="?", default="Hello from Pocket.")
    args = parser.parse_args(argv)

    with PocketPipeline.load(args.bundle_dir, precision=args.precision) as pipeline:
        pipeline.set_default_voice(args.voice)
        result = pipeline(args.text)
        result.save_wav(args.output)

    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

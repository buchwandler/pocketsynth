from __future__ import annotations

import argparse
from pathlib import Path

from . import PocketPipeline, __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pocketsynth")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    synth = sub.add_parser("synthesize")
    source = synth.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle")
    source.add_argument("--bundle-dir", type=Path)
    synth.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    synth.add_argument("--voice", type=Path, required=True, help="mono 16-bit PCM WAV prompt")
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("text")
    args = parser.parse_args(argv)

    if args.command == "synthesize":
        pipeline = (
            PocketPipeline.from_pretrained(args.bundle, precision=args.precision)
            if args.bundle
            else PocketPipeline.load(args.bundle_dir, precision=args.precision)
        )
        with pipeline:
            pipeline.set_default_voice(args.voice)
            result = pipeline.run(args.text)
            result.save_wav(args.output)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

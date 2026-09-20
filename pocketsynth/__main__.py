from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import PocketPipeline, __version__
from ._onnxvoice import normalize_pocket_ref
from .bundle import BundlePaths
from .config import GenerationConfig
from .voice import _read_pcm_wav


def _providers(values: list[str] | None) -> str | list[str] | None:
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lsd-steps", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frames-after-eos", type=int)
    parser.add_argument("--normalize-audio", action="store_true")
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--provider", action="append", dest="providers")


def _synthesize(args: argparse.Namespace) -> int:
    generation = GenerationConfig(
        temperature=args.temperature,
        lsd_steps=args.lsd_steps,
        max_frames=args.max_frames,
        frames_after_eos=args.frames_after_eos,
        normalize_audio=args.normalize_audio,
        volume=args.volume,
    )
    providers = _providers(args.providers)
    pipeline_kwargs: dict[str, Any] = {
        "precision": args.precision,
        "generation": generation,
        "providers": providers,
    }
    if args.bundle:
        pipeline_kwargs.update(
            cache_dir=args.cache_dir,
            offline=args.offline,
        )
        pipeline = PocketPipeline.from_pretrained(args.bundle, **pipeline_kwargs)
    else:
        pipeline = PocketPipeline.load(args.bundle_dir, **pipeline_kwargs)
    with pipeline:
        pipeline.set_default_voice(args.voice)
        pipeline.run(args.text).save_wav(args.output)
    return 0


def _check(args: argparse.Namespace) -> int:
    failures = 0
    try:
        import onnxvoice

        print(f"OnnxVoice: {onnxvoice.__version__}")
    except Exception as exc:
        print(f"FAIL OnnxVoice import: {exc}")
        failures += 1
        onnxvoice = None

    available: tuple[str, ...] = ()
    if onnxvoice is not None:
        try:
            available = tuple(onnxvoice.available_providers())
            print(f"ORT providers: {', '.join(available)}")
        except Exception as exc:
            print(f"FAIL ORT provider discovery: {exc}")
            failures += 1

    providers = _providers(args.providers)
    requested = [providers] if isinstance(providers, str) else providers or []
    missing = [provider for provider in requested if provider not in available]
    if missing:
        print(f"FAIL requested ORT providers unavailable: {', '.join(missing)}")
        failures += 1

    if args.bundle_dir:
        try:
            paths = BundlePaths.from_directory(args.bundle_dir, precision=args.precision)
            print(f"Local bundle: OK ({paths.root}, sample rate {paths.metadata.sample_rate})")
        except Exception as exc:
            print(f"FAIL local bundle: {exc}")
            failures += 1
    elif args.bundle:
        if onnxvoice is None:
            failures += 1
        else:
            try:
                manager = onnxvoice.OnnxVoice(cache_dir=args.cache_dir, offline=args.offline)
                manager.resolve(normalize_pocket_ref(args.bundle), quality=args.precision)
                print(f"Catalog: OK ({args.bundle})")
            except Exception as exc:
                print(f"FAIL catalog bundle {args.bundle!r}: {exc}")
                failures += 1
    else:
        print("Bundle: not checked (pass --bundle or --bundle-dir)")

    if args.voice:
        try:
            _, sample_rate = _read_pcm_wav(args.voice)
            print(f"Voice WAV: OK (mono PCM16, sample rate {sample_rate})")
        except Exception as exc:
            print(f"FAIL voice WAV: {exc}")
            failures += 1
    else:
        print("Voice WAV: not checked (pass --voice)")

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pocketsynth")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synthesize")
    source = synth.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle")
    source.add_argument("--bundle-dir", type=Path)
    _add_runtime_options(synth)
    synth.add_argument("--voice", type=Path, required=True, help="mono 16-bit PCM WAV prompt")
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--cache-dir", type=Path)
    synth.add_argument("--offline", action="store_true")
    synth.add_argument("text")

    check = sub.add_parser("check", help="check dependencies, providers, bundles, and voice WAVs")
    check_source = check.add_mutually_exclusive_group()
    check_source.add_argument("--bundle")
    check_source.add_argument("--bundle-dir", type=Path)
    check.add_argument("--voice", type=Path)
    check.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    check.add_argument("--cache-dir", type=Path)
    check.add_argument("--offline", action="store_true")
    check.add_argument("--provider", action="append", dest="providers")

    args = parser.parse_args(argv)
    if args.command == "synthesize":
        return _synthesize(args)
    if args.command == "check":
        return _check(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

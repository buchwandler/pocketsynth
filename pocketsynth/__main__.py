from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from . import (
    PocketRuntime,
    __version__,
    discover_bundles,
    inspect_voice_prompt,
    list_voice_prompts,
)
from .bundle import BundlePaths
from .config import GenerationConfig
from .convenience import synthesize_with_runtime
from .errors import PocketSynthError
from .voice import _looks_like_path_string, _read_pcm_wav


def _providers(values: list[str] | None) -> str | list[str] | None:
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--lsd-steps", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frames-after-eos", type=int)
    parser.add_argument("--provider", action="append", dest="providers")


def _synthesize(args: argparse.Namespace) -> int:
    generation = GenerationConfig(
        temperature=args.temperature,
        lsd_steps=args.lsd_steps,
        max_frames=args.max_frames,
        frames_after_eos=args.frames_after_eos,
    )
    providers = _providers(args.providers)
    runtime_options: dict[str, Any] = {
        "precision": args.precision,
        "providers": providers,
    }
    if args.bundle:
        runtime_options.update(
            cache_dir=args.cache_dir,
            offline=args.offline,
            refresh_catalog=args.refresh_catalog,
            force_download=args.force_download,
        )
    try:
        if args.bundle:
            runtime = PocketRuntime.from_pretrained(args.bundle, **runtime_options)
            bundle_label = args.bundle
        else:
            runtime = PocketRuntime.load(args.bundle_dir, **runtime_options)
            bundle_label = str(args.bundle_dir)
        with runtime as active_runtime:
            result = synthesize_with_runtime(
                active_runtime,
                args.text,
                voice=args.voice,
                generation=generation,
                sentence_split=args.sentence_split,
            )
            result.save_wav(args.output)
    except PocketSynthError as exc:
        print(f"FAIL synthesis: {exc}", file=sys.stderr)
        return 1
    print(f"Output: {args.output}")
    print(f"Bundle: {bundle_label}")
    print(f"Precision: {args.precision}")
    print(f"Sample rate: {result.sample_rate} Hz")
    print(f"Duration: {result.duration_seconds:.3f} s")
    return 0


def _voices_list(args: argparse.Namespace) -> int:
    try:
        prompts = list_voice_prompts(
            cache_dir=args.cache_dir,
            offline=args.offline,
            dataset=args.dataset,
            variant=args.variant,
            license=args.license,
        )
    except Exception as exc:
        print(f"FAIL voice prompt catalog: {exc}", file=sys.stderr)
        return 1
    print("REF\tDATASET\tVARIANT\tLICENSE\tSOURCE_PATH")
    for prompt in prompts:
        print(
            f"{prompt.ref}\t{prompt.dataset}\t{prompt.variant}\t"
            f"{prompt.license}\t{prompt.source_path}"
        )
    return 0


def _check(args: argparse.Namespace) -> int:
    failures = 0
    onnxvoice: Any = None
    try:
        onnxvoice = import_module("onnxvoice")
        print(f"OnnxVoice: {onnxvoice.__version__}")
    except Exception as exc:
        print(f"FAIL OnnxVoice import: {exc}")
        failures += 1

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

    available_voices: tuple[str, ...] | None = None
    if args.bundle_dir:
        try:
            paths = BundlePaths.from_directory(args.bundle_dir, precision=args.precision)
            available_voices = paths.metadata.predefined_voices
            print(f"Local bundle: OK ({paths.root}, sample rate {paths.metadata.sample_rate})")
        except Exception as exc:
            print(f"FAIL local bundle: {exc}")
            failures += 1
    elif args.bundle:
        try:
            bundles = discover_bundles(cache_dir=args.cache_dir, offline=args.offline)
            bundle = next(
                (item for item in bundles if args.bundle in (item.id, item.ref, *item.aliases)),
                None,
            )
            if bundle is None:
                raise ValueError(f"Unknown Pocket bundle {args.bundle!r}")
            if args.precision not in bundle.precisions:
                raise ValueError(
                    f"Bundle {bundle.id!r} does not provide precision {args.precision!r}"
                )
            available_voices = bundle.predefined_voices
            print(f"Catalog: OK ({bundle.ref})")
        except Exception as exc:
            print(f"FAIL catalog bundle {args.bundle!r}: {exc}")
            failures += 1
    else:
        print("Bundle: not checked (pass --bundle or --bundle-dir)")

    if args.voice:
        if args.voice.startswith("kyutai-tts-voices:"):
            try:
                prompt = inspect_voice_prompt(
                    args.voice, cache_dir=args.cache_dir, offline=args.offline
                )
                print(f"Managed voice prompt: OK ({prompt.ref})")
            except Exception as exc:
                print(f"FAIL managed voice prompt {args.voice!r}: {exc}")
                failures += 1
        elif available_voices is not None and args.voice in available_voices:
            print(f"Predefined voice: OK ({args.voice})")
        elif _looks_like_path_string(args.voice):
            try:
                _, sample_rate = _read_pcm_wav(args.voice)
                print(f"Voice WAV: OK (sample rate {sample_rate} Hz)")
            except Exception as exc:
                print(f"FAIL voice WAV: {exc}")
                failures += 1
        elif available_voices is not None:
            names = ", ".join(available_voices) or "none"
            print(
                f"FAIL predefined voice {args.voice!r} is not declared by the bundle; "
                f"available voices: {names}"
            )
            failures += 1
        elif args.bundle or args.bundle_dir:
            print("Voice: not checked (bundle metadata unavailable)")
        else:
            print(
                "Voice: not checked (pass --bundle or --bundle-dir to validate a predefined name)"
            )
    else:
        print("Voice: not checked (pass --voice)")

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
    synth.add_argument(
        "--sentence-split",
        choices=("phrasplit", "none"),
        default="none",
        help=(
            'Optional sentence segmentation before Pocket model-limit chunking. "none" (default) '
            'does not segment sentences; "phrasplit" uses Phrasplit\'s lightweight regex '
            "backend (no spaCy). Model-limit chunking still applies in either mode."
        ),
    )
    synth.add_argument(
        "--voice",
        required=True,
        help="bundle-declared voice name, kyutai-tts-voices:<id>, or local WAV path",
    )
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--cache-dir", type=Path)
    synth.add_argument("--offline", action="store_true")
    synth.add_argument("--refresh-catalog", action="store_true")
    synth.add_argument("--force-download", action="store_true")
    synth.add_argument("text")

    check = sub.add_parser("check", help="check dependencies, providers, bundles, and voice inputs")
    check_source = check.add_mutually_exclusive_group()
    check_source.add_argument("--bundle")
    check_source.add_argument("--bundle-dir", type=Path)
    check.add_argument("--voice", help="predefined name, kyutai-tts-voices:<id>, or local WAV path")
    check.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    check.add_argument("--cache-dir", type=Path)
    check.add_argument("--offline", action="store_true")
    check.add_argument("--provider", action="append", dest="providers")

    voices = sub.add_parser("voices", help="discover managed voice prompts")
    voices_sub = voices.add_subparsers(dest="voices_command", required=True)
    voices_list = voices_sub.add_parser(
        "list", help="list cataloged prompts without downloading audio"
    )
    voices_list.add_argument("--cache-dir", type=Path)
    voices_list.add_argument("--offline", action="store_true")
    voices_list.add_argument("--dataset")
    voices_list.add_argument("--variant")
    voices_list.add_argument("--license")

    args = parser.parse_args(argv)
    if args.command == "synthesize":
        return _synthesize(args)
    if args.command == "check":
        return _check(args)
    if args.command == "voices":
        return _voices_list(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

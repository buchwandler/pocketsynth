"""Measure and compare Pocket TTS semantic shape and model-token lengths."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    from ._output import artifact_path
except ImportError:  # Support direct execution as `python examples/token_shape_matrix.py`.
    from _output import artifact_path

from pocketsynth import PocketRuntime
from pocketsynth.audio import write_wav
from pocketsynth.config import DEFAULT_TEMPERATURE, GenerationConfig
from pocketsynth.convenience import synthesize_with_runtime
from pocketsynth.errors import SynthesisInputTooLongError
from pocketsynth.text_split import SentenceSplitMode
from pocketsynth.types import RenderedSegment, SynthesisRequest
from pocketsynth.voice import PreparedVoice

SemanticShape = Literal["one_sentence", "multiple_sentences"]

_DEFAULT_TARGETS = (10, 20, 30, 40, 45, 50)
_SEMANTIC_SHAPES: tuple[SemanticShape, ...] = ("one_sentence", "multiple_sentences")
_SPLIT_MODES: tuple[SentenceSplitMode, ...] = ("none", "phrasplit")
_CSV_FIELDS = (
    "case_id",
    "semantic_shape",
    "target_tokens",
    "actual_tokens",
    "token_delta",
    "maximum_tokens",
    "atomic_fits",
    "sentence_split",
    "chunk_count",
    "chunk_token_counts",
    "duration_seconds",
    "rms",
    "peak",
    "onset_zcr",
    "onset_spectral_centroid_hz",
    "onset_hf_ratio_ge_4khz",
    "temperature",
    "lsd_steps",
    "frames_after_eos",
    "atomic_status",
    "atomic_wav_path",
    "wav_path",
    "text",
)

_ONE_SENTENCE_HEAD = "The narrator spoke clearly"
_ONE_SENTENCE_CLAUSES = (
    "while the quiet audience listened",
    "as the lights dimmed around the room",
    "and the speaker continued at an even pace",
    "while everyone followed the explanation carefully",
    "because each detail mattered to the final result",
    "and the discussion remained calm and focused",
    "as the main idea became easier to understand",
    "while the steady rhythm carried the story forward",
)
_MULTI_SENTENCES = (
    "We spoke.",
    "They listened.",
    "The room stayed quiet.",
    "The speaker continued calmly.",
    "Everyone followed the explanation.",
    "The final point was easy to understand.",
)


@dataclass(frozen=True, slots=True)
class MatrixText:
    semantic_shape: SemanticShape
    target_tokens: int
    actual_tokens: int
    maximum_tokens: int
    text: str

    @property
    def token_delta(self) -> int:
        return self.actual_tokens - self.target_tokens

    @property
    def atomic_fits(self) -> bool:
        return self.actual_tokens <= self.maximum_tokens


@dataclass(frozen=True, slots=True)
class MatrixResult:
    case_id: str
    semantic_shape: SemanticShape
    target_tokens: int
    actual_tokens: int
    token_delta: int
    maximum_tokens: int
    atomic_fits: bool
    sentence_split: SentenceSplitMode
    chunk_count: int
    chunk_token_counts: tuple[int, ...]
    duration_seconds: float | None
    rms: float | None
    peak: float | None
    onset_zcr: float | None
    onset_spectral_centroid_hz: float | None
    onset_hf_ratio_ge_4khz: float | None
    temperature: float
    lsd_steps: int
    frames_after_eos: int | None
    atomic_status: str
    atomic_wav_path: str | None
    wav_path: str | None
    text: str


def _parse_targets(value: str) -> tuple[int, ...]:
    try:
        targets = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "targets must be comma-separated positive integers"
        ) from exc
    if not targets or any(target <= 0 for target in targets):
        raise argparse.ArgumentTypeError("targets must be comma-separated positive integers")
    if len(set(targets)) != len(targets):
        raise argparse.ArgumentTypeError("targets must not contain duplicates")
    return targets


def _matrix_targets(requested: Iterable[int], *, maximum: int, over_by: int) -> tuple[int, ...]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if over_by < 1:
        raise ValueError("over_by must be positive")
    return tuple(dict.fromkeys((*requested, maximum + over_by)))


def _measure_text(runtime: PocketRuntime, *, case_id: str, text: str) -> tuple[int, int]:
    measure = runtime.measure_request(
        SynthesisRequest(id=case_id, text=text, language=runtime.bundle_language)
    )
    if measure.maximum is None:
        raise RuntimeError("Pocket runtime did not report a hard token maximum")
    return measure.amount, measure.maximum


def _candidate_texts(semantic_shape: SemanticShape) -> Iterable[str]:
    if semantic_shape == "one_sentence":
        yield f"{_ONE_SENTENCE_HEAD}."
        for clause_count in range(1, 33):
            for offset in range(len(_ONE_SENTENCE_CLAUSES)):
                clauses = tuple(
                    _ONE_SENTENCE_CLAUSES[(offset + index) % len(_ONE_SENTENCE_CLAUSES)]
                    for index in range(clause_count)
                )
                yield f"{_ONE_SENTENCE_HEAD}, {', '.join(clauses)}."
        return

    for sentence_count in range(2, 49):
        for offset in range(len(_MULTI_SENTENCES)):
            sentences = tuple(
                _MULTI_SENTENCES[(offset + index) % len(_MULTI_SENTENCES)]
                for index in range(sentence_count)
            )
            yield " ".join(sentences)


def _find_candidate_for_target(
    runtime: PocketRuntime,
    *,
    semantic_shape: SemanticShape,
    target_tokens: int,
    tolerance: int = 1,
) -> MatrixText:
    if semantic_shape not in _SEMANTIC_SHAPES:
        raise ValueError(f"unsupported semantic shape: {semantic_shape}")
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must not be negative")

    best: tuple[tuple[int, int, int, str], MatrixText] | None = None
    measured: dict[str, tuple[int, int]] = {}
    for candidate in _candidate_texts(semantic_shape):
        if candidate in measured:
            continue
        actual_tokens, maximum_tokens = _measure_text(
            runtime,
            case_id=f"candidate-{semantic_shape}-{target_tokens}",
            text=candidate,
        )
        measured[candidate] = (actual_tokens, maximum_tokens)
        matrix_text = MatrixText(
            semantic_shape=semantic_shape,
            target_tokens=target_tokens,
            actual_tokens=actual_tokens,
            maximum_tokens=maximum_tokens,
            text=candidate,
        )
        rank = (
            abs(actual_tokens - target_tokens),
            int(actual_tokens > target_tokens),
            len(candidate),
            candidate,
        )
        if best is None or rank < best[0]:
            best = (rank, matrix_text)
        if actual_tokens == target_tokens:
            best = (rank, matrix_text)
            break

    if best is None:
        raise RuntimeError(f"no candidate text generated for {semantic_shape}")
    selected = best[1]
    if selected.actual_tokens != target_tokens:
        tolerance_note = "within" if abs(selected.token_delta) <= tolerance else "outside"
        print(
            f"Warning: {semantic_shape} target {target_tokens} measured "
            f"{selected.actual_tokens} tokens (delta {selected.token_delta}; "
            f"{tolerance_note} tolerance {tolerance}).",
            file=sys.stderr,
        )
    return selected


def _rms(audio: np.ndarray) -> float:
    signal = np.asarray(audio, dtype=np.float64)
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal))))


def _onset_metrics(
    audio: np.ndarray,
    sample_rate: int,
    seconds: float = 1.0,
) -> dict[str, float]:
    signal = np.asarray(audio, dtype=np.float64)
    if sample_rate <= 0 or seconds <= 0:
        raise ValueError("sample_rate and seconds must be positive")
    onset = signal[: min(signal.size, max(1, int(sample_rate * seconds)))]
    if onset.size == 0:
        return {
            "onset_zcr": 0.0,
            "onset_spectral_centroid_hz": 0.0,
            "onset_hf_ratio_ge_4khz": 0.0,
        }

    crossings = np.count_nonzero(np.signbit(onset[1:]) != np.signbit(onset[:-1]))
    zcr = float(crossings / max(onset.size - 1, 1))
    windowed = onset * np.hanning(onset.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(onset.size, d=1.0 / sample_rate)
    magnitude_sum = float(np.sum(spectrum))
    centroid = float(np.sum(frequencies * spectrum) / magnitude_sum) if magnitude_sum > 0 else 0.0
    energy = np.square(spectrum)
    energy_sum = float(np.sum(energy))
    high_frequency_ratio = (
        float(np.sum(energy[frequencies >= 4000.0]) / energy_sum) if energy_sum > 0 else 0.0
    )
    return {
        "onset_zcr": zcr,
        "onset_spectral_centroid_hz": centroid,
        "onset_hf_ratio_ge_4khz": high_frequency_ratio,
    }


def _shape_directory(shape: SemanticShape) -> str:
    return "one-sentence" if shape == "one_sentence" else "multiple-sentences"


def _target_label(target: int, over_target: int, maximum: int) -> str:
    return f">{maximum}" if target == over_target else str(target)


def _effective_temperature(runtime: PocketRuntime, requested: float | None) -> tuple[float, str]:
    if requested is not None:
        return requested, "explicit"
    recommended = runtime.metadata.default_temperature
    if recommended is not None:
        return recommended, "bundle"
    return DEFAULT_TEMPERATURE, "pocketsynth_default"


def _atomic_check(
    runtime: PocketRuntime,
    case: MatrixText,
    *,
    case_id: str,
    voice: PreparedVoice | None,
    generation: GenerationConfig,
    output_dir: Path,
    render_atomic: bool,
    measure_only: bool,
) -> tuple[str, str | None]:
    if not case.atomic_fits:
        if measure_only or voice is None:
            return "over_limit_not_rendered_measure_only", None
        try:
            runtime.synthesize(
                SynthesisRequest(id=case_id, text=case.text, language=runtime.bundle_language),
                voice=voice,
                config=generation,
            )
        except SynthesisInputTooLongError:
            return "expected_too_long", None
        raise AssertionError("strict synthesis unexpectedly accepted an over-limit request")

    if not render_atomic or measure_only or voice is None:
        return "fits_not_rendered", None
    strict_result = runtime.synthesize(
        SynthesisRequest(id=case_id, text=case.text, language=runtime.bundle_language),
        voice=voice,
        config=generation,
    )
    relative_path = Path(_shape_directory(case.semantic_shape)) / (
        f"target-{case.target_tokens:03d}__actual-{case.actual_tokens:03d}__atomic.wav"
    )
    path = output_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, strict_result.audio, strict_result.sample_rate)
    return "rendered", relative_path.as_posix()


def _render_result(
    runtime: PocketRuntime,
    case: MatrixText,
    *,
    case_id: str,
    sentence_split: SentenceSplitMode,
    voice: PreparedVoice | None,
    generation: GenerationConfig,
    temperature: float,
    atomic_status: str,
    atomic_wav_path: str | None,
    output_dir: Path,
    measure_only: bool,
) -> MatrixResult:
    if measure_only or voice is None:
        return MatrixResult(
            case_id=case_id,
            semantic_shape=case.semantic_shape,
            target_tokens=case.target_tokens,
            actual_tokens=case.actual_tokens,
            token_delta=case.token_delta,
            maximum_tokens=case.maximum_tokens,
            atomic_fits=case.atomic_fits,
            sentence_split=sentence_split,
            chunk_count=0,
            chunk_token_counts=(),
            duration_seconds=None,
            rms=None,
            peak=None,
            onset_zcr=None,
            onset_spectral_centroid_hz=None,
            onset_hf_ratio_ge_4khz=None,
            temperature=temperature,
            lsd_steps=generation.lsd_steps,
            frames_after_eos=generation.frames_after_eos,
            atomic_status=atomic_status,
            atomic_wav_path=atomic_wav_path,
            wav_path=None,
            text=case.text,
        )

    rendered: RenderedSegment = synthesize_with_runtime(
        runtime,
        case.text,
        voice=voice,
        sentence_split=sentence_split,
        generation=generation,
        id=case_id,
    )
    relative_path = Path(_shape_directory(case.semantic_shape)) / (
        f"target-{case.target_tokens:03d}__actual-{case.actual_tokens:03d}__{sentence_split}.wav"
    )
    path = output_dir / relative_path
    rendered.save_wav(path)
    audio = np.asarray(rendered.audio, dtype=np.float64)
    onset = _onset_metrics(audio, rendered.sample_rate)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    effective: Mapping[str, object] = {}
    if rendered.chunks:
        chunk_metadata = rendered.chunks[0].metadata
        raw_effective = chunk_metadata.get("effective_generation", {})
        if isinstance(raw_effective, Mapping):
            effective = raw_effective
    frames_after_eos = effective.get("frames_after_eos", generation.frames_after_eos)
    return MatrixResult(
        case_id=case_id,
        semantic_shape=case.semantic_shape,
        target_tokens=case.target_tokens,
        actual_tokens=case.actual_tokens,
        token_delta=case.token_delta,
        maximum_tokens=case.maximum_tokens,
        atomic_fits=case.atomic_fits,
        sentence_split=sentence_split,
        chunk_count=len(rendered.chunks),
        chunk_token_counts=tuple(len(chunk.token_ids) for chunk in rendered.chunks),
        duration_seconds=float(audio.size / rendered.sample_rate),
        rms=_rms(audio),
        peak=peak,
        onset_zcr=onset["onset_zcr"],
        onset_spectral_centroid_hz=onset["onset_spectral_centroid_hz"],
        onset_hf_ratio_ge_4khz=onset["onset_hf_ratio_ge_4khz"],
        temperature=temperature,
        lsd_steps=generation.lsd_steps,
        frames_after_eos=(int(frames_after_eos) if frames_after_eos is not None else None),
        atomic_status=atomic_status,
        atomic_wav_path=atomic_wav_path,
        wav_path=relative_path.as_posix(),
        text=case.text,
    )


def _write_outputs(
    output_dir: Path,
    *,
    runtime: PocketRuntime,
    voice: str,
    targets: tuple[int, ...],
    over_target: int,
    cases: tuple[MatrixText, ...],
    results: tuple[MatrixResult, ...],
    generation: GenerationConfig,
    temperature: float,
    temperature_source: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["chunk_token_counts"] = ";".join(str(count) for count in result.chunk_token_counts)
            writer.writerow(row)

    text_records = [
        {
            "case_id": f"{case.semantic_shape}__target-{case.target_tokens:03d}",
            "semantic_shape": case.semantic_shape,
            "target_tokens": case.target_tokens,
            "actual_tokens": case.actual_tokens,
            "token_delta": case.token_delta,
            "maximum_tokens": case.maximum_tokens,
            "atomic_fits": case.atomic_fits,
            "text": case.text,
        }
        for case in cases
    ]
    (output_dir / "texts.json").write_text(
        json.dumps(text_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    results_by_case: dict[str, list[MatrixResult]] = {}
    for result in results:
        results_by_case.setdefault(result.case_id, []).append(result)
    case_records = []
    for case in cases:
        case_id = f"{case.semantic_shape}__target-{case.target_tokens:03d}"
        case_results = results_by_case.get(case_id, [])
        atomic_status = case_results[0].atomic_status if case_results else "not_run"
        atomic_wav_path = case_results[0].atomic_wav_path if case_results else None
        case_records.append(
            {
                "case_id": case_id,
                "semantic_shape": case.semantic_shape,
                "target_tokens": case.target_tokens,
                "actual_tokens": case.actual_tokens,
                "token_delta": case.token_delta,
                "maximum_tokens": case.maximum_tokens,
                "atomic_fits": case.atomic_fits,
                "atomic_status": atomic_status,
                "atomic_wav_path": atomic_wav_path,
                "text": case.text,
                "renders": [asdict(result) for result in case_results],
            }
        )
    manifest = {
        "schema": "pocketsynth.example.token-shape-matrix.v1",
        "bundle": runtime.bundle_id,
        "bundle_language": runtime.bundle_language,
        "bundle_revision": runtime.source_revision,
        "precision": runtime.precision,
        "maximum_tokens": cases[0].maximum_tokens
        if cases
        else runtime.metadata.max_token_per_chunk,
        "voice": voice,
        "generation": {
            "temperature": temperature,
            "temperature_source": temperature_source,
            "lsd_steps": generation.lsd_steps,
            "max_frames": generation.max_frames,
            "frames_after_eos": generation.frames_after_eos,
        },
        "targets": list(targets),
        "over_target": over_target,
        "over_target_label": f">{runtime.metadata.max_token_per_chunk}",
        "cases": case_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(
        output_dir / "summary.md",
        cases=cases,
        results=results,
        over_target=over_target,
        maximum=runtime.metadata.max_token_per_chunk,
        temperature=temperature,
        temperature_source=temperature_source,
        lsd_steps=generation.lsd_steps,
    )


def _write_summary(
    path: Path,
    *,
    cases: tuple[MatrixText, ...],
    results: tuple[MatrixResult, ...],
    over_target: int,
    maximum: int,
    temperature: float,
    temperature_source: str,
    lsd_steps: int,
) -> None:
    result_lookup = {(result.case_id, result.sentence_split): result for result in results}
    target_labels = tuple(
        dict.fromkeys(_target_label(case.target_tokens, over_target, maximum) for case in cases)
    )
    lines = [
        "# PocketSynth token-shape matrix",
        "",
        f"Hard maximum: {maximum} model tokens.",
        f"Temperature: {temperature} ({temperature_source}); LSD steps: {lsd_steps}.",
        "",
        "Target token counts are requested buckets.",
        "Actual token counts come from PocketRuntime.measure_request().",
        f"Requested target buckets: {', '.join(target_labels)}.",
        f"The >{maximum} bucket resolves to {over_target} model tokens.",
        "",
        "## Cases",
        "",
        "| Shape | Target | Actual | Fits atomically | none chunks | phrasplit chunks |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for case in cases:
        case_id = f"{case.semantic_shape}__target-{case.target_tokens:03d}"
        none = result_lookup.get((case_id, "none"))
        sentence = result_lookup.get((case_id, "phrasplit"))
        none_count = str(none.chunk_count) if none and none.chunk_count else "—"
        sentence_count = str(sentence.chunk_count) if sentence and sentence.chunk_count else "—"
        lines.append(
            f"| {case.semantic_shape} | "
            f"{_target_label(case.target_tokens, over_target, maximum)} | {case.actual_tokens} | "
            f"{'yes' if case.atomic_fits else 'no'} | {none_count} | {sentence_count} |"
        )

    lines.extend(
        [
            "",
            "## Render comparison",
            "",
            "| Shape | Tokens | Policy | Duration (s) | Onset ZCR | Onset HF >=4kHz | WAV |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for result in results:
        duration = f"{result.duration_seconds:.3f}" if result.duration_seconds is not None else "—"
        zcr = f"{result.onset_zcr:.5f}" if result.onset_zcr is not None else "—"
        hf = (
            f"{result.onset_hf_ratio_ge_4khz:.5f}"
            if result.onset_hf_ratio_ge_4khz is not None
            else "—"
        )
        wav = result.wav_path or "—"
        lines.append(
            f"| {result.semantic_shape} | {result.actual_tokens} | {result.sentence_split} | "
            f"{duration} | {zcr} | {hf} | {wav} |"
        )

    lines.extend(["", "## Texts", ""])
    for case in cases:
        lines.extend(
            [
                f"### {case.semantic_shape} — target {_target_label(case.target_tokens, over_target, maximum)}, "
                f"actual {case.actual_tokens}",
                "",
                "```text",
                case.text,
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_console_summary(
    *,
    runtime: PocketRuntime,
    voice: str,
    cases: tuple[MatrixText, ...],
    results: tuple[MatrixResult, ...],
    over_target: int,
    temperature: float,
    temperature_source: str,
    output_dir: Path,
) -> None:
    print(f"Bundle: {runtime.bundle_id}")
    print(f"Hard maximum: {runtime.metadata.max_token_per_chunk} tokens")
    print(f"Voice: {voice}")
    print(f"Temperature: {temperature} ({temperature_source})")
    print()
    print("SHAPE                 TARGET  ACTUAL  FITS   NONE CHUNKS  SENTENCE CHUNKS")
    result_lookup = {(result.case_id, result.sentence_split): result for result in results}
    maximum = runtime.metadata.max_token_per_chunk
    for case in cases:
        case_id = f"{case.semantic_shape}__target-{case.target_tokens:03d}"
        none = result_lookup.get((case_id, "none"))
        phrase = result_lookup.get((case_id, "phrasplit"))
        none_count = str(none.chunk_count) if none and none.chunk_count else "—"
        phrase_count = str(phrase.chunk_count) if phrase and phrase.chunk_count else "—"
        print(
            f"{case.semantic_shape:<21} "
            f"{_target_label(case.target_tokens, over_target, maximum):>6} "
            f"{case.actual_tokens:>7}  {'yes' if case.atomic_fits else 'no':<5} "
            f"{none_count:>12} {phrase_count:>16}"
        )
    print()
    print(f"Artifacts: {output_dir}")
    print(f"Summary:   {output_dir / 'summary.md'}")
    print(f"Manifest:  {output_dir / 'manifest.csv'}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare semantic sentence shape and active Pocket model-token lengths "
            "with paired explicit sentence-splitting policies."
        )
    )
    parser.add_argument("--bundle", default="english_2026-04")
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--targets", type=_parse_targets, default=_DEFAULT_TARGETS)
    parser.add_argument("--over-by", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--lsd-steps", type=int, default=1)
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--require-exact-targets", action="store_true")
    parser.add_argument("--render-atomic", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.over_by < 1:
        parser.error("--over-by must be positive")
    if args.lsd_steps < 1:
        parser.error("--lsd-steps must be positive")
    try:
        generation = GenerationConfig(
            temperature=args.temperature,
            lsd_steps=args.lsd_steps,
        )
    except ValueError as exc:
        parser.error(str(exc))

    offline = args.offline or os.environ.get("POCKETSYNTH_EXAMPLE_OFFLINE") == "1"
    output_dir = (
        artifact_path("token-shape-matrix")
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    with PocketRuntime.from_pretrained(
        args.bundle,
        precision=args.precision,
        offline=offline,
    ) as runtime:
        maximum = runtime.metadata.max_token_per_chunk
        if maximum < 1:
            raise RuntimeError("active bundle reported a non-positive token maximum")
        over_target = maximum + args.over_by
        targets = _matrix_targets(args.targets, maximum=maximum, over_by=args.over_by)
        cases = tuple(
            _find_candidate_for_target(
                runtime,
                semantic_shape=shape,
                target_tokens=target,
            )
            for shape in _SEMANTIC_SHAPES
            for target in targets
        )
        missed = [case for case in cases if case.actual_tokens != case.target_tokens]
        for case in missed:
            print(
                f"Target not exact: {case.semantic_shape} target {case.target_tokens}, "
                f"actual {case.actual_tokens}.",
                file=sys.stderr,
            )

        temperature, temperature_source = _effective_temperature(runtime, args.temperature)

        prepared_voice = None if args.measure_only else runtime.prepare_voice(args.voice)
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[MatrixResult] = []
        for case in cases:
            case_id = f"{case.semantic_shape}__target-{case.target_tokens:03d}"
            atomic_status, atomic_wav_path = _atomic_check(
                runtime,
                case,
                case_id=case_id,
                voice=prepared_voice,
                generation=generation,
                output_dir=output_dir,
                render_atomic=args.render_atomic,
                measure_only=args.measure_only,
            )
            for split_mode in _SPLIT_MODES:
                results.append(
                    _render_result(
                        runtime,
                        case,
                        case_id=case_id,
                        sentence_split=split_mode,
                        voice=prepared_voice,
                        generation=generation,
                        temperature=temperature,
                        atomic_status=atomic_status,
                        atomic_wav_path=atomic_wav_path,
                        output_dir=output_dir,
                        measure_only=args.measure_only,
                    )
                )

        result_tuple = tuple(results)
        _write_outputs(
            output_dir,
            runtime=runtime,
            voice=args.voice,
            targets=targets,
            over_target=over_target,
            cases=cases,
            results=result_tuple,
            generation=generation,
            temperature=temperature,
            temperature_source=temperature_source,
        )
        _print_console_summary(
            runtime=runtime,
            voice=args.voice,
            cases=cases,
            results=result_tuple,
            over_target=over_target,
            temperature=temperature,
            temperature_source=temperature_source,
            output_dir=output_dir,
        )

    if args.require_exact_targets and missed:
        formatted = ", ".join(
            f"{case.semantic_shape} {case.target_tokens}->{case.actual_tokens}" for case in missed
        )
        print(f"Exact token targets were not found: {formatted}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

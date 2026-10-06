from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from examples import token_shape_matrix as matrix
from pocketsynth.config import GenerationConfig
from pocketsynth.errors import SynthesisInputTooLongError
from pocketsynth.types import RenderedChunk, RenderedSegment, SynthesisRequest


class MeasuringRuntime:
    bundle_language = "en"

    def __init__(self, counts: dict[str, int], *, maximum: int = 50) -> None:
        self.counts = counts
        self.maximum = maximum
        self.requests: list[SynthesisRequest] = []

    def measure_request(self, request: SynthesisRequest) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(
            amount=self.counts.get(request.text, len(request.text.split())),
            maximum=self.maximum,
        )


def _fixed_candidate(
    _runtime: object,
    *,
    semantic_shape: matrix.SemanticShape,
    target_tokens: int,
    tolerance: int = 1,
) -> matrix.MatrixText:
    del tolerance
    return matrix.MatrixText(
        semantic_shape=semantic_shape,
        target_tokens=target_tokens,
        actual_tokens=target_tokens,
        maximum_tokens=50,
        text=f"{semantic_shape} sample at target {target_tokens}.",
    )


def _fake_runtime_context() -> tuple[MagicMock, MagicMock]:
    runtime = MagicMock()
    runtime.metadata = SimpleNamespace(max_token_per_chunk=50, default_temperature=0.3)
    runtime.bundle_id = "english-test"
    runtime.bundle_language = "en"
    runtime.source_revision = "revision-test"
    runtime.precision = "int8"
    context = MagicMock()
    context.__enter__.return_value = runtime
    context.__exit__.return_value = False
    return runtime, context


def _rendered_segment(text: str, request_id: str) -> RenderedSegment:
    audio = np.sin(np.arange(1200, dtype=np.float32) * 0.08)
    chunk = RenderedChunk(
        index=0,
        text=text,
        model_text=text,
        token_ids=(1, 2, 3),
        audio=audio,
        sample_rate=24_000,
        metadata={"effective_generation": {"temperature": 0.3, "frames_after_eos": 3}},
    )
    return RenderedSegment(
        id=request_id,
        audio=audio,
        sample_rate=24_000,
        text=text,
        language="en",
        token_ids=(1, 2, 3),
        chunks=(chunk,),
    )


def test_matrix_targets_include_requested_buckets_and_dynamic_over_maximum() -> None:
    assert matrix._matrix_targets(matrix._DEFAULT_TARGETS, maximum=50, over_by=5) == (
        10,
        20,
        30,
        40,
        45,
        50,
        55,
    )
    assert matrix._matrix_targets((10, 20), maximum=64, over_by=5) == (10, 20, 69)


def test_candidate_builders_preserve_the_requested_semantic_shape() -> None:
    one_sentence = next(iter(matrix._candidate_texts("one_sentence")))
    multiple_sentences = next(iter(matrix._candidate_texts("multiple_sentences")))

    assert one_sentence.endswith(".")
    assert one_sentence.count(".") == 1
    assert multiple_sentences.count(".") >= 2


def test_candidate_search_uses_runtime_measurement_and_prefers_exact_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = ("first candidate", "exact candidate", "unused candidate")
    monkeypatch.setattr(matrix, "_candidate_texts", lambda _shape: iter(candidates))
    runtime = MeasuringRuntime({"first candidate": 8, "exact candidate": 4})

    case = matrix._find_candidate_for_target(
        cast(matrix.PocketRuntime, runtime), semantic_shape="one_sentence", target_tokens=4
    )

    assert case.text == "exact candidate"
    assert case.target_tokens == 4
    assert case.actual_tokens == 4
    assert case.token_delta == 0
    assert case.atomic_fits
    assert [request.text for request in runtime.requests] == list(candidates[:2])


def test_candidate_search_records_nearest_measured_count_and_prefers_undershoot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    candidates = ("overshooting candidate", "undershooting candidate")
    monkeypatch.setattr(matrix, "_candidate_texts", lambda _shape: iter(candidates))
    runtime = MeasuringRuntime({candidates[0]: 8, candidates[1]: 4})

    case = matrix._find_candidate_for_target(
        cast(matrix.PocketRuntime, runtime), semantic_shape="multiple_sentences", target_tokens=6
    )

    assert case.text == "undershooting candidate"
    assert case.target_tokens == 6
    assert case.actual_tokens == 4
    assert case.token_delta == -2
    assert "measured 4 tokens" in capsys.readouterr().err


def test_onset_metrics_are_finite_and_handle_empty_audio() -> None:
    audio = np.sin(np.arange(24_000, dtype=np.float32) * 0.1)

    metrics = matrix._onset_metrics(audio, sample_rate=24_000)
    empty_metrics = matrix._onset_metrics(np.zeros(0), sample_rate=24_000)

    assert metrics["onset_zcr"] > 0
    assert metrics["onset_spectral_centroid_hz"] > 0
    assert 0.0 <= metrics["onset_hf_ratio_ge_4khz"] <= 1.0
    assert set(empty_metrics.values()) == {0.0}


def test_main_renders_both_split_modes_reuses_voice_and_records_atomic_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, context = _fake_runtime_context()
    prepared_voice = object()
    runtime.prepare_voice.return_value = prepared_voice
    runtime.synthesize.side_effect = SynthesisInputTooLongError(
        text_length=100, token_count=55, max_tokens=50, bundle_id="english-test"
    )
    monkeypatch.setattr(matrix.PocketRuntime, "from_pretrained", MagicMock(return_value=context))
    monkeypatch.setattr(matrix, "_find_candidate_for_target", _fixed_candidate)
    render_calls: list[tuple[object, object, str, GenerationConfig]] = []

    def render(
        active_runtime: object,
        text: str,
        *,
        voice: object,
        sentence_split: matrix.SentenceSplitMode,
        generation: GenerationConfig,
        id: str,
    ) -> RenderedSegment:
        render_calls.append((active_runtime, voice, sentence_split, generation))
        return _rendered_segment(text, id)

    monkeypatch.setattr(matrix, "synthesize_with_runtime", render)
    output_dir = tmp_path / "rendered"

    status = matrix.main(["--targets", "10", "--over-by", "5", "--output-dir", str(output_dir)])

    assert status == 0
    runtime.prepare_voice.assert_called_once_with("alba")
    assert len(render_calls) == 8
    assert {call[0] for call in render_calls} == {runtime}
    assert {call[1] for call in render_calls} == {prepared_voice}
    assert [call[2] for call in render_calls].count("none") == 4
    assert [call[2] for call in render_calls].count("phrasplit") == 4
    assert runtime.synthesize.call_count == 2
    assert (output_dir / "one-sentence/target-010__actual-010__none.wav").is_file()
    assert (output_dir / "one-sentence/target-010__actual-010__phrasplit.wav").is_file()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "pocketsynth.example.token-shape-matrix.v1"
    assert manifest["targets"] == [10, 55]
    assert manifest["over_target_label"] == ">50"
    assert manifest["generation"]["temperature"] == 0.3
    over_limit_cases = [case for case in manifest["cases"] if case["target_tokens"] == 55]
    assert len(over_limit_cases) == 2
    assert all(case["atomic_status"] == "expected_too_long" for case in over_limit_cases)
    with (output_dir / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {row["sentence_split"] for row in rows} == {"none", "phrasplit"}
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Target token counts are requested buckets." in summary
    assert "Actual token counts come from PocketRuntime.measure_request()." in summary
    assert "The >50 bucket resolves to 55 model tokens." in summary
    assert "## Render comparison" in summary


def test_measure_only_outputs_are_stable_and_exact_target_misses_fail_after_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime, context = _fake_runtime_context()
    monkeypatch.setattr(matrix.PocketRuntime, "from_pretrained", MagicMock(return_value=context))
    monkeypatch.setattr(matrix, "_find_candidate_for_target", _fixed_candidate)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    assert matrix.main(["--targets", "10", "--measure-only", "--output-dir", str(first_dir)]) == 0
    assert matrix.main(["--targets", "10", "--measure-only", "--output-dir", str(second_dir)]) == 0
    runtime.prepare_voice.assert_not_called()

    for filename in ("manifest.csv", "manifest.json", "texts.json", "summary.md"):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()

    def missed_candidate(
        _runtime: object,
        *,
        semantic_shape: matrix.SemanticShape,
        target_tokens: int,
        tolerance: int = 1,
    ) -> matrix.MatrixText:
        del tolerance
        return matrix.MatrixText(
            semantic_shape=semantic_shape,
            target_tokens=target_tokens,
            actual_tokens=target_tokens + 1,
            maximum_tokens=50,
            text=f"{semantic_shape} missed bucket {target_tokens}.",
        )

    monkeypatch.setattr(matrix, "_find_candidate_for_target", missed_candidate)
    failed_dir = tmp_path / "exact-required"
    status = matrix.main(
        [
            "--targets",
            "10",
            "--measure-only",
            "--require-exact-targets",
            "--output-dir",
            str(failed_dir),
        ]
    )

    assert status == 1
    assert (failed_dir / "manifest.json").is_file()
    stderr = capsys.readouterr().err
    assert "one_sentence 10->11" in stderr
    assert "multiple_sentences 55->56" in stderr


def test_atomic_over_limit_is_expected_and_optional_fit_render_writes_wav(
    tmp_path: Path,
) -> None:
    over_limit = matrix.MatrixText("one_sentence", 6, 6, 5, "This sentence is too long.")
    runtime = MagicMock()
    runtime.bundle_language = "en"
    runtime.synthesize.side_effect = SynthesisInputTooLongError(
        text_length=25, token_count=6, max_tokens=5, bundle_id="english-test"
    )

    status, wav_path = matrix._atomic_check(
        runtime,
        over_limit,
        case_id="too-long",
        voice=cast(matrix.PreparedVoice, object()),
        generation=GenerationConfig(),
        output_dir=tmp_path,
        render_atomic=True,
        measure_only=False,
    )

    assert status == "expected_too_long"
    assert wav_path is None

    fitting = matrix.MatrixText("one_sentence", 3, 3, 5, "We spoke.")
    runtime.synthesize.side_effect = None
    runtime.synthesize.return_value = SimpleNamespace(
        audio=np.zeros(64, dtype=np.float32), sample_rate=24_000
    )
    status, wav_path = matrix._atomic_check(
        runtime,
        fitting,
        case_id="fits",
        voice=cast(matrix.PreparedVoice, object()),
        generation=GenerationConfig(),
        output_dir=tmp_path,
        render_atomic=True,
        measure_only=False,
    )

    assert status == "rendered"
    assert wav_path is not None
    assert (tmp_path / wav_path).is_file()

"""Demonstrate explicit sentence splitting in convenience rendering."""

from __future__ import annotations

import argparse
import os

from _output import artifact_path

from pocketsynth import PocketRuntime
from pocketsynth.convenience import synthesize_with_runtime
from pocketsynth.text_split import split_text_for_synthesis
from pocketsynth.types import SynthesisRequest

TEXT = (
    "Dr. Smith arrived early. He reviewed the notes carefully. "
    "Then he started the presentation. The audience listened closely."
)
BUNDLE = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04")
OFFLINE = os.environ.get("POCKETSYNTH_EXAMPLE_OFFLINE") == "1"


def _print_measure(runtime: PocketRuntime, *, request_id: str, label: str, text: str) -> None:
    measure = runtime.measure_request(
        SynthesisRequest(id=request_id, text=text, language=runtime.bundle_language)
    )
    maximum = str(measure.maximum) if measure.maximum is not None else "unbounded"
    print(f"{label}: {measure.amount} / {maximum} tokens")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate opt-in lightweight sentence segmentation and the "
            "sentence_split='none' bypass."
        )
    )
    parser.parse_args()

    split_output = artifact_path("long-text.wav")
    unsplit_output = artifact_path("long-text-no-split.wav")
    with PocketRuntime.from_pretrained(BUNDLE, offline=OFFLINE) as runtime:
        _print_measure(runtime, request_id="whole-text", label="Whole text", text=TEXT)
        sentences = split_text_for_synthesis(
            TEXT, language=runtime.bundle_language, mode="phrasplit"
        )
        for index, sentence in enumerate(sentences):
            _print_measure(
                runtime,
                request_id=f"sentence-{index}",
                label=f"Sentence {index}",
                text=sentence,
            )
        result = synthesize_with_runtime(runtime, TEXT, voice="alba", sentence_split="phrasplit")
        result.save_wav(split_output)
        for chunk in result.chunks:
            print(chunk.index, chunk.text)

        unsplit_result = synthesize_with_runtime(runtime, TEXT, voice="alba", sentence_split="none")
        unsplit_result.save_wav(unsplit_output)

    print(f"Sentence-split WAV: {split_output}")
    print(f"Model-limit-only WAV: {unsplit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

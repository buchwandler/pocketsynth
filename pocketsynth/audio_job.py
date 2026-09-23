from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from audiocompose import AudioJob

from .audio import postprocess_audio
from .config import GenerationConfig
from .plan_adapter import PreparedPocketSpan, PreparedPocketUnit
from .voice import PreparedVoice


@dataclass(frozen=True, slots=True)
class RenderedPocketSpan:
    span: PreparedPocketSpan
    audio: np.ndarray
    sample_rate: int
    token_ids: tuple[int, ...]
    metadata: Mapping[str, Any]


def render_span(
    span: PreparedPocketSpan,
    *,
    runtime: Any,
    voice: PreparedVoice,
    generation: GenerationConfig,
) -> RenderedPocketSpan:
    parts: list[np.ndarray] = []
    token_ids: list[int] = []
    model_chunks = runtime.frontend.split_for_model(span.text)
    for chunk_text in model_chunks:
        ids = runtime.frontend.encode(chunk_text)
        token_ids.extend(ids)
        audio = runtime.infer_tokens(ids, voice, generation)
        if audio.size:
            parts.append(audio)
    combined = (
        np.concatenate(parts).astype(np.float32, copy=False)
        if parts
        else np.zeros(0, dtype=np.float32)
    )
    combined = postprocess_audio(
        combined, normalize=generation.normalize_audio, volume=generation.volume
    )
    return RenderedPocketSpan(
        span=span,
        audio=combined,
        sample_rate=runtime.sample_rate,
        token_ids=tuple(token_ids),
        metadata={"model_chunks": list(model_chunks)},
    )


def build_audio_job(
    *,
    plan: Any,
    prepared_units: Sequence[PreparedPocketUnit],
    runtime: Any,
    default_voice: PreparedVoice,
    voice_bindings: Mapping[str, PreparedVoice],
    generation: GenerationConfig,
    producer_version: str,
) -> tuple[AudioJob, tuple[RenderedPocketSpan, ...]]:
    """Create an AudioCompose job plus rendered-span metadata.

    Import AudioCompose lazily so package metadata/frontend tests do not need the optional runtime.
    """
    from audiocompose import (
        AudioBufferSource,
        AudioClip,
        AudioJob,
        LoudnessPolicy,
        OutputPolicy,
        Silence,
    )

    items: list[Any] = []
    rendered: list[RenderedPocketSpan] = []
    for unit in prepared_units:
        for span in unit.spans:
            voice = (
                voice_bindings.get(span.voice_ref, default_voice)
                if span.voice_ref
                else default_voice
            )
            if span.pause_before_seconds > 0:
                items.append(
                    Silence(
                        f"pause:{span.id}:before",
                        span.pause_before_seconds,
                        {"pocketsynth.span_id": span.id, "pocketsynth.position": "before"},
                    )
                )
            result = render_span(span, runtime=runtime, voice=voice, generation=generation)
            rendered.append(result)
            if result.audio.size:
                items.append(
                    AudioClip(
                        span.id,
                        AudioBufferSource(result.audio, result.sample_rate),
                        metadata={
                            "pocketsynth.span_id": span.id,
                            "pocketsynth.segment_ids": list(span.segment_ids),
                            "pocketsynth.token_ids": list(result.token_ids),
                        },
                    )
                )
            if span.pause_after_seconds > 0:
                items.append(
                    Silence(
                        f"pause:{span.id}:after",
                        span.pause_after_seconds,
                        {"pocketsynth.span_id": span.id, "pocketsynth.position": "after"},
                    )
                )
    output = OutputPolicy(
        sample_rate=runtime.sample_rate,
        channels=1,
        loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
        clip_policy="clamp",
    )
    job = AudioJob(
        tuple(items),
        output=output,
        producer={"name": "pocketsynth", "version": producer_version},
        source={"utterplan.plan_id": plan.plan_id, "utterplan.schema_version": plan.schema_version},
    )
    return job, tuple(rendered)

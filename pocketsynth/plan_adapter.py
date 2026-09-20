from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from utterplan import PlanSegment, UtterancePlan, normalize_language

from .errors import (
    UnsupportedPlanDirectiveError,
    UnsupportedPlanLanguageError,
    VoiceBindingError,
)


@dataclass(frozen=True, slots=True)
class PreparedPocketSpan:
    id: str
    text: str
    language: str
    segment_ids: tuple[str, ...]
    spoken_start: int
    spoken_end: int
    pause_before_seconds: float
    pause_after_seconds: float
    voice_ref: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedPocketUnit:
    plan_unit_id: str
    index: int
    kind: Literal["paragraph", "sentence"]
    spoken_start: int
    spoken_end: int
    segment_ids: tuple[str, ...]
    marker_ids: tuple[str, ...]
    content_hash: str
    spans: tuple[PreparedPocketSpan, ...]


def _voice_ref(segment: PlanSegment) -> str | None:
    voice = getattr(segment.directives, "voice", None)
    return getattr(voice, "reference", None) if voice is not None else None


def _unsupported(segment: PlanSegment) -> list[str]:
    result: list[str] = []
    directives = segment.directives
    for name in ("pronunciation", "prosody", "emphasis", "audio"):
        if getattr(directives, name, None) is not None:
            result.append(name)
    return result


def _issue(policy: str, message: str) -> str | None:
    if policy == "error":
        raise UnsupportedPlanDirectiveError(message)
    return message if policy == "warn" else None


def _compatible(a: PlanSegment, b: PlanSegment) -> bool:
    return (
        normalize_language(a.language) == normalize_language(b.language)
        and _voice_ref(a) == _voice_ref(b)
        and a.pause_after.seconds == 0
        and b.pause_before.seconds == 0
        and not _unsupported(a)
        and not _unsupported(b)
    )


def prepare_plan(
    plan: UtterancePlan,
    *,
    bundle_language: str,
    directive_policy: str = "error",
    language_policy: str = "strict",
    voice_bindings: Mapping[str, Any] | None = None,
) -> tuple[PreparedPocketUnit, ...]:
    plan.validate()
    segments = {segment.id: segment for segment in plan.segments}
    expected = normalize_language(bundle_language)
    prepared_units: list[PreparedPocketUnit] = []
    for unit in plan.units:
        unit_segments = [segments[segment_id] for segment_id in unit.segment_ids]
        spans: list[PreparedPocketSpan] = []
        group: list[PlanSegment] = []

        def flush(
            current_group: list[PlanSegment],
            current_spans: list[PreparedPocketSpan],
            current_unit: Any,
        ) -> None:
            if not current_group:
                return
            first, last = current_group[0], current_group[-1]
            warnings: list[str] = []
            for segment in current_group:
                language = normalize_language(segment.language)
                if language_policy == "strict" and language != expected:
                    raise UnsupportedPlanLanguageError(
                        f"Pocket bundle language {expected!r} cannot render {language!r}"
                    )
                unsupported = _unsupported(segment)
                if unsupported:
                    warning = _issue(
                        directive_policy,
                        f"Pocket MVP does not implement directives {unsupported} on {segment.id}",
                    )
                    if warning:
                        warnings.append(warning)
            voice_ref = _voice_ref(first)
            if voice_ref and voice_bindings is not None and voice_ref not in voice_bindings:
                raise VoiceBindingError(f"No Pocket voice binding for {voice_ref!r}")
            current_spans.append(
                PreparedPocketSpan(
                    id=f"{current_unit.id}:span-{len(current_spans):03d}",
                    text=plan.texts.spoken[first.spoken_start : last.spoken_end],
                    language=first.language,
                    segment_ids=tuple(s.id for s in current_group),
                    spoken_start=first.spoken_start,
                    spoken_end=last.spoken_end,
                    pause_before_seconds=float(first.pause_before.seconds),
                    pause_after_seconds=float(last.pause_after.seconds),
                    voice_ref=voice_ref,
                    warnings=tuple(warnings),
                )
            )
            current_group.clear()

        for segment in unit_segments:
            if group and not _compatible(group[-1], segment):
                flush(group, spans, unit)
            group.append(segment)
            if segment.pause_after.seconds > 0 or _unsupported(segment):
                flush(group, spans, unit)
        flush(group, spans, unit)
        prepared_units.append(
            PreparedPocketUnit(
                plan_unit_id=unit.id,
                index=unit.index,
                kind=unit.kind,
                spoken_start=unit.spoken_start,
                spoken_end=unit.spoken_end,
                segment_ids=tuple(unit.segment_ids),
                marker_ids=tuple(unit.marker_ids),
                content_hash=unit.content_hash,
                spans=tuple(spans),
            )
        )
    return tuple(prepared_units)

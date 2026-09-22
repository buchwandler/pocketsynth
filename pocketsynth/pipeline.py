from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from utterplan import UtterancePlan, UtterancePlanner

from ._onnxvoice import install_pretrained_bundle
from .audio import silence_samples
from .audio_job import build_audio_job, render_span
from .bundle import BundleMetadata, BundlePaths, Precision
from .config import GenerationConfig, PipelineConfig
from .diagnostics import TimingDiagnostics
from .errors import PipelineClosedError, VoiceBindingError
from .plan_adapter import PreparedPocketUnit
from .plan_adapter import prepare_plan as adapt_plan
from .planning import planner_config_from_pocketsynth, planner_language
from .runtime import PocketRuntime
from .types import AudioChunk, AudioResult, AudioUnitDescriptor, AudioUnitResult
from .voice import PreparedVoice

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"


class PreparedAudioUnits:
    def __init__(
        self,
        pipeline: PocketPipeline,
        plan: UtterancePlan,
        units: tuple[PreparedPocketUnit, ...],
        default_voice: PreparedVoice,
        voice_bindings: Mapping[str, PreparedVoice],
        generation: GenerationConfig,
    ) -> None:
        self.pipeline = pipeline
        self.plan = plan
        self._units = units
        self.default_voice = default_voice
        self.voice_bindings = dict(voice_bindings)
        self.generation = generation
        self.descriptors = tuple(
            AudioUnitDescriptor(
                index=u.index,
                unit_kind=u.kind,
                text=plan.texts.spoken[u.spoken_start : u.spoken_end],
                char_start=u.spoken_start,
                char_end=u.spoken_end,
                plan_unit_id=u.plan_unit_id,
                content_hash=u.content_hash,
                segment_ids=u.segment_ids,
                marker_ids=u.marker_ids,
            )
            for u in units
        )

    def render(self) -> Iterator[AudioUnitResult]:
        by_index = {d.index: d for d in self.descriptors}
        for unit in self._units:
            parts: list[np.ndarray] = []
            ids: list[int] = []
            warnings: list[str] = []
            for span in unit.spans:
                voice = (
                    self.voice_bindings.get(span.voice_ref, self.default_voice)
                    if span.voice_ref
                    else self.default_voice
                )
                if span.voice_ref and span.voice_ref not in self.voice_bindings:
                    raise VoiceBindingError(f"No Pocket voice binding for {span.voice_ref!r}")
                before = silence_samples(
                    self.pipeline.runtime.sample_rate, span.pause_before_seconds
                )
                if before:
                    parts.append(np.zeros(before, dtype=np.float32))
                rendered = render_span(
                    span,
                    runtime=self.pipeline.runtime,
                    voice=voice,
                    generation=self.generation,
                )
                ids.extend(rendered.token_ids)
                warnings.extend(span.warnings)
                if rendered.audio.size:
                    parts.append(rendered.audio)
                after = silence_samples(self.pipeline.runtime.sample_rate, span.pause_after_seconds)
                if after:
                    parts.append(np.zeros(after, dtype=np.float32))
            audio = (
                np.concatenate(parts).astype(np.float32, copy=False)
                if parts
                else np.zeros(0, dtype=np.float32)
            )
            yield AudioUnitResult(
                descriptor=by_index[unit.index],
                audio=audio,
                sample_rate=self.pipeline.runtime.sample_rate,
                token_ids=tuple(ids),
                warnings=tuple(warnings),
                metadata={"plan_id": self.plan.plan_id},
            )


class PocketPipeline:
    """Reusable UtterPlan semantic planner and Pocket bundle renderer."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        runtime: PocketRuntime | None = None,
        planner: UtterancePlanner | None = None,
    ) -> None:
        self.config = config
        self._runtime = runtime
        self._planner = planner
        self._closed = False
        self._last_planning_ms: float | None = None
        self._default_voice: PreparedVoice | None = None
        self._bundle_metadata = (
            runtime.metadata
            if runtime is not None
            else BundleMetadata.load(config.bundle_dir / "bundle.json")
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        precision: Precision = "int8",
        generation: GenerationConfig | None = None,
        providers: Any | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        language: str | None = None,
        unit: Literal["paragraph", "sentence"] = "sentence",
        text_preparation: Literal["identity", "spokenform"] = "spokenform",
        document_format: Literal["plain", "ssmd"] = "plain",
        pauses: Any | None = None,
        linguistics: Any | None = None,
        ssmd: Any | None = None,
        overlap_mode: Literal["snap", "strict"] = "snap",
        language_aliases: Mapping[str, str] | None = None,
        planner_diagnostics: bool = True,
        directive_policy: Literal["error", "warn", "ignore"] = "error",
        language_policy: Literal["strict", "allow"] = "strict",
        retain_unit_audio: bool = False,
        return_diagnostics: bool = True,
    ) -> PocketPipeline:
        """Open a local Pocket bundle directory without network access."""
        paths = BundlePaths.from_directory(directory, precision=precision)
        runtime = PocketRuntime.load(
            directory,
            precision=precision,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        config = PipelineConfig(
            bundle_dir=paths.root,
            precision=precision,
            generation=generation or GenerationConfig(),
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            language=language or planner_language(paths.metadata),
            unit=unit,
            text_preparation=text_preparation,
            document_format=document_format,
            pauses=pauses,
            linguistics=linguistics,
            ssmd=ssmd,
            overlap_mode=overlap_mode,
            language_aliases=language_aliases or {},
            planner_diagnostics=planner_diagnostics,
            directive_policy=directive_policy,
            language_policy=language_policy,
            retain_unit_audio=retain_unit_audio,
            return_diagnostics=return_diagnostics,
        )
        return cls(config, runtime=runtime)

    @classmethod
    def from_pretrained(
        cls,
        bundle: str,
        *,
        precision: Precision = "int8",
        cache_dir: str | Path | None = None,
        offline: bool | None = None,
        refresh_catalog: bool = False,
        force_download: bool = False,
        generation: GenerationConfig | None = None,
        providers: Any | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        language: str | None = None,
        unit: Literal["paragraph", "sentence"] = "sentence",
        text_preparation: Literal["identity", "spokenform"] = "spokenform",
        progress: Any | None = None,
        document_format: Literal["plain", "ssmd"] = "plain",
        pauses: Any | None = None,
        linguistics: Any | None = None,
        ssmd: Any | None = None,
        overlap_mode: Literal["snap", "strict"] = "snap",
        language_aliases: Mapping[str, str] | None = None,
        planner_diagnostics: bool = True,
        directive_policy: Literal["error", "warn", "ignore"] = "error",
        language_policy: Literal["strict", "allow"] = "strict",
        retain_unit_audio: bool = False,
        return_diagnostics: bool = True,
    ) -> PocketPipeline:
        """Open a managed Pocket bundle, downloading if needed."""
        resolved = install_pretrained_bundle(
            bundle,
            precision=precision,
            cache_dir=cache_dir,
            offline=offline,
            refresh_catalog=refresh_catalog,
            force_download=force_download,
            progress=progress,
        )
        runtime = PocketRuntime.from_resolved(
            resolved,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        config = PipelineConfig(
            bundle_dir=resolved.path,
            precision=precision,
            generation=generation or GenerationConfig(),
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            language=language or planner_language(runtime.metadata),
            unit=unit,
            text_preparation=text_preparation,
            document_format=document_format,
            pauses=pauses,
            linguistics=linguistics,
            ssmd=ssmd,
            overlap_mode=overlap_mode,
            language_aliases=language_aliases or {},
            planner_diagnostics=planner_diagnostics,
            directive_policy=directive_policy,
            language_policy=language_policy,
            retain_unit_audio=retain_unit_audio,
            return_diagnostics=return_diagnostics,
        )
        return cls(config, runtime=runtime)

    @property
    def runtime(self) -> PocketRuntime:
        self._ensure_open()
        if self._runtime is None:
            self._runtime = PocketRuntime.load(
                self.config.bundle_dir,
                precision=self.config.precision,
                providers=self.config.providers,
                provider_options=self.config.provider_options,
                session_options=self.config.session_options,
            )
        return self._runtime

    def _planner_instance(self) -> UtterancePlanner:
        if self._planner is None:
            self._planner = UtterancePlanner(
                planner_config_from_pocketsynth(self.config, self._bundle_metadata)
            )
        return self._planner

    def plan(
        self,
        text: str,
        *,
        unit: Literal["paragraph", "sentence"] | None = None,
    ) -> UtterancePlan:
        self._ensure_open()
        started = time.perf_counter()
        planner_config = planner_config_from_pocketsynth(
            self.config, self._bundle_metadata, unit=unit
        )
        result = self._planner_instance().plan(text, config=planner_config, unit=unit)
        self._last_planning_ms = (time.perf_counter() - started) * 1000
        return result

    def prepare_voice(self, source: Any) -> PreparedVoice:
        if isinstance(source, PreparedVoice):
            source.validate_compatible(
                bundle_id=self.runtime.bundle_id,
                sample_rate=self.runtime.sample_rate,
            )
            return source
        return self.runtime.prepare_voice(source)

    def set_default_voice(self, source: Any) -> PreparedVoice:
        self._default_voice = self.prepare_voice(source)
        return self._default_voice

    def prepare_plan(
        self,
        plan: UtterancePlan,
        *,
        voice: PreparedVoice | Any | None = None,
        voice_bindings: Mapping[str, PreparedVoice | Any] | None = None,
    ) -> PreparedAudioUnits:
        default_voice = self._resolve_voice(voice)
        bindings: dict[str, PreparedVoice] = {}
        for name, value in dict(voice_bindings or {}).items():
            bindings[name] = (
                value if isinstance(value, PreparedVoice) else self.prepare_voice(value)
            )
        units = adapt_plan(
            plan,
            bundle_language=planner_language(self._bundle_metadata),
            directive_policy=self.config.directive_policy,
            language_policy=self.config.language_policy,
            voice_bindings=bindings,
        )
        return PreparedAudioUnits(
            self, plan, units, default_voice, bindings, self.config.generation
        )

    def to_audio_job(
        self,
        plan: UtterancePlan,
        *,
        voice: PreparedVoice | Any | None = None,
        voice_bindings: Mapping[str, PreparedVoice | Any] | None = None,
    ):
        default_voice = self._resolve_voice(voice)
        bindings = {
            name: value if isinstance(value, PreparedVoice) else self.prepare_voice(value)
            for name, value in dict(voice_bindings or {}).items()
        }
        units = adapt_plan(
            plan,
            bundle_language=planner_language(self._bundle_metadata),
            directive_policy=self.config.directive_policy,
            language_policy=self.config.language_policy,
            voice_bindings=bindings,
        )
        job, _ = build_audio_job(
            plan=plan,
            prepared_units=units,
            runtime=self.runtime,
            default_voice=default_voice,
            voice_bindings=bindings,
            generation=self.config.generation,
            producer_version=__version__,
        )
        return job

    def render_plan(
        self,
        plan: UtterancePlan,
        *,
        voice: PreparedVoice | Any | None = None,
        voice_bindings: Mapping[str, PreparedVoice | Any] | None = None,
    ) -> AudioResult:
        from audiocompose import Composer

        self._ensure_open()
        started = time.perf_counter()
        default_voice = self._resolve_voice(voice)
        bindings = {
            name: value if isinstance(value, PreparedVoice) else self.prepare_voice(value)
            for name, value in dict(voice_bindings or {}).items()
        }
        units = adapt_plan(
            plan,
            bundle_language=planner_language(self._bundle_metadata),
            directive_policy=self.config.directive_policy,
            language_policy=self.config.language_policy,
            voice_bindings=bindings,
        )
        inference_started = time.perf_counter()
        job, rendered = build_audio_job(
            plan=plan,
            prepared_units=units,
            runtime=self.runtime,
            default_voice=default_voice,
            voice_bindings=bindings,
            generation=self.config.generation,
            producer_version=__version__,
        )
        inference_ms = (time.perf_counter() - inference_started) * 1000
        compose_started = time.perf_counter()
        composition = Composer().compose(job)
        composition_ms = (time.perf_counter() - compose_started) * 1000
        chunks = (
            [
                AudioChunk(
                    sample_rate=item.sample_rate,
                    audio=item.audio,
                    token_ids=item.token_ids,
                    warnings=item.span.warnings,
                    metadata=dict(item.metadata),
                )
                for item in rendered
            ]
            if self.config.retain_unit_audio
            else []
        )
        diagnostics = self.runtime.diagnostics if self.config.return_diagnostics else None
        if diagnostics is not None:
            diagnostics = replace(
                diagnostics,
                plan_id=plan.plan_id,
                utterplan_producer=dict(plan.producer),
                utterplan_schema_version=plan.schema_version,
            )
        timing = (
            TimingDiagnostics(
                planning_ms=self._last_planning_ms,
                inference_ms=inference_ms,
                composition_ms=composition_ms,
                total_ms=(time.perf_counter() - started) * 1000,
            )
            if self.config.return_diagnostics
            else None
        )
        return AudioResult(
            audio=composition.audio,
            sample_rate=composition.sample_rate,
            source_text=plan.source.text,
            prepared_text=plan.texts.spoken,
            plan=plan,
            plan_id=plan.plan_id,
            chunks=chunks,
            warnings=tuple(plan.warnings)
            + tuple(w for item in rendered for w in item.span.warnings),
            diagnostics=diagnostics,
            timing=timing,
            metadata={
                "bundle_id": self.runtime.bundle_id,
                "precision": self.runtime.precision,
                "source_revision": self.runtime.source_revision,
                "sample_rate": composition.sample_rate,
                "plan_id": plan.plan_id,
                "model_chunk_count": len(rendered),
            },
        )

    def run(
        self,
        text: str,
        *,
        voice: Any | None = None,
        voice_bindings: Mapping[str, Any] | None = None,
    ) -> AudioResult:
        """Plan text and render to audio."""
        plan = self.plan(text)
        return self.render_plan(plan, voice=voice, voice_bindings=voice_bindings)

    def __call__(self, text: str, *, voice: Any | None = None, **kwargs: Any) -> AudioResult:
        return self.run(text, voice=voice, **kwargs)

    def iter_units(
        self,
        text: str,
        *,
        voice: Any | None = None,
        voice_bindings: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioUnitResult]:
        plan = self.plan(text, unit="sentence")
        yield from self.prepare_plan(plan, voice=voice, voice_bindings=voice_bindings).render()

    def _resolve_voice(self, voice: PreparedVoice | Any | None) -> PreparedVoice:
        if voice is None:
            if self._default_voice is None:
                raise VoiceBindingError(
                    "A Pocket voice prompt is required. Call set_default_voice(), pass voice=, "
                    "or provide voice bindings."
                )
            return self._default_voice
        return voice if isinstance(voice, PreparedVoice) else self.prepare_voice(voice)

    def close(self) -> None:
        if self._closed:
            return
        if self._runtime is not None:
            self._runtime.close()
        if self._planner is not None:
            self._planner.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise PipelineClosedError("PocketPipeline is closed")

    def __enter__(self) -> PocketPipeline:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

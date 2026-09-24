from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ._onnxvoice import (
    ResolvedPocketBundle,
    _call,
    install_pretrained_bundle,
    open_installed_bundle,
    open_local_bundle,
    runtime_diagnostics,
)
from .asset_progress import AssetProgressCallback
from .bundle import BundleMetadata, BundlePaths, Precision
from .config import GenerationConfig
from .diagnostics import RuntimeDiagnostics, SynthesisTiming
from .errors import (
    BundleLanguageError,
    ModelInferenceError,
    RuntimeClosedError,
    UnsupportedBundleError,
)
from .frontend import PocketFrontend
from .language import bundle_language as resolve_bundle_language
from .language import normalize_language
from .types import RenderedChunk, RenderedSegment, SynthesisSegment
from .voice import PreparedVoice, prepare_voice


class PocketRuntime:
    """Low-level Pocket bundle runtime backed by an OnnxVoice PocketAdapter."""

    def __init__(
        self,
        *,
        paths: BundlePaths | None,
        metadata: BundleMetadata,
        tokenizer_path: Path,
        runtime: Any,
        bundle_id: str | None = None,
        precision: str | None = None,
        source_revision: str | None = None,
    ) -> None:
        self.paths = paths
        self.metadata = metadata
        self.tokenizer_path = tokenizer_path
        self.runtime = runtime
        self.bundle_id = bundle_id or metadata.bundle_name
        self.precision = precision
        self.source_revision = source_revision
        self.frontend = PocketFrontend(tokenizer_path, metadata)
        self._closed = False

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        precision: Precision = "int8",
        providers: Sequence[Any] | str | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
    ) -> PocketRuntime:  # noqa: UP037
        paths = BundlePaths.from_directory(directory, precision=precision)
        runtime = open_local_bundle(
            paths,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        return cls(
            paths=paths,
            metadata=paths.metadata,
            tokenizer_path=paths.tokenizer,
            runtime=runtime,
            precision=precision,
        )

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
        providers: Sequence[Any] | str | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        progress: AssetProgressCallback | None = None,
    ) -> PocketRuntime:
        resolved = install_pretrained_bundle(
            bundle,
            precision=precision,
            cache_dir=cache_dir,
            offline=offline,
            refresh_catalog=refresh_catalog,
            force_download=force_download,
            progress=progress,
        )
        return cls.from_resolved(
            resolved,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            cache_dir=cache_dir,
            offline=bool(offline),
        )

    @classmethod
    def from_resolved(
        cls,
        resolved: ResolvedPocketBundle,
        *,
        providers: Sequence[Any] | str | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        cache_dir: str | Path | None = None,
        offline: bool = False,
    ) -> PocketRuntime:  # noqa: UP037
        metadata = BundleMetadata.load(resolved.metadata_path)
        raw_voice_names = resolved.metadata.get("predefined_voice_names")
        if raw_voice_names is not None:
            if not isinstance(raw_voice_names, Sequence) or isinstance(
                raw_voice_names, (str, bytes)
            ):
                raise UnsupportedBundleError(
                    "Pocket catalog predefined_voice_names must be a sequence"
                )
            catalog_voice_names = tuple(raw_voice_names)
            if not all(isinstance(name, str) for name in catalog_voice_names):
                raise UnsupportedBundleError(
                    "Pocket catalog predefined_voice_names must contain strings"
                )
            if metadata.predefined_voices and set(metadata.predefined_voices) != set(
                catalog_voice_names
            ):
                raise UnsupportedBundleError(
                    "Pocket bundle and catalog disagree about predefined voice names"
                )
            if not metadata.predefined_voices:
                metadata = replace(metadata, predefined_voices=catalog_voice_names)
        runtime = open_installed_bundle(
            resolved,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            cache_dir=cache_dir,
            offline=offline,
        )
        return cls(
            paths=None,
            metadata=metadata,
            tokenizer_path=resolved.tokenizer_path,
            runtime=runtime,
            bundle_id=resolved.bundle_id,
            precision=resolved.precision,
            source_revision=resolved.source_revision,
        )

    @property
    def predefined_voices(self) -> tuple[str, ...]:
        return self.metadata.predefined_voices

    @property
    def bundle_language(self) -> str:
        return resolve_bundle_language(self.metadata)

    @property
    def sample_rate(self) -> int:
        return self.metadata.sample_rate

    def prepare_voice(self, source: Any) -> PreparedVoice:
        self._ensure_open()

        def load_voice() -> PreparedVoice:
            return prepare_voice(
                self.runtime,
                source,
                sample_rate=self.sample_rate,
                bundle_id=self.bundle_id,
                predefined_voices=self.predefined_voices,
            )

        if isinstance(source, str) and source in self.predefined_voices:
            voice = _call("prepare_voice", load_voice)
        else:
            voice = load_voice()
        voice.validate_compatible(bundle_id=self.bundle_id, sample_rate=self.sample_rate)
        if voice.bundle_id == self.bundle_id:
            return voice
        return PreparedVoice(
            state=voice.state,
            sample_rate=voice.sample_rate,
            source=voice.source,
            bundle_id=self.bundle_id,
            runtime_fingerprint=self.bundle_id,
            metadata=voice.metadata,
            fingerprint=voice.fingerprint,
        )

    def infer_tokens(
        self,
        token_ids: Sequence[int],
        voice: PreparedVoice,
        generation: GenerationConfig,
    ) -> np.ndarray:
        self._ensure_open()
        voice.validate_compatible(bundle_id=self.bundle_id, sample_rate=self.sample_rate)
        if len(token_ids) > self.metadata.max_token_per_chunk:
            raise ValueError(
                f"Pocket token sequence has {len(token_ids)} tokens; "
                f"maximum is {self.metadata.max_token_per_chunk}"
            )
        try:
            result = self.runtime.infer(
                token_ids,
                voice_state=voice.state,
                temperature=generation.temperature,
                lsd_steps=generation.lsd_steps,
                max_frames=generation.max_frames,
                frames_after_eos=(
                    generation.frames_after_eos
                    if generation.frames_after_eos is not None
                    else self.metadata.model_recommended_frames_after_eos
                ),
            )
        except Exception as exc:
            if isinstance(exc, (ValueError, ModelInferenceError)):
                raise
            raise ModelInferenceError(f"Pocket ONNX inference failed: {exc}") from exc
        sample_rate = int(getattr(result, "sample_rate", 0))
        if sample_rate != self.sample_rate:
            raise ModelInferenceError(
                f"runtime sample rate {sample_rate} does not match bundle {self.sample_rate}"
            )
        audio = np.asarray(result.audio, dtype=np.float32)
        if audio.ndim != 1 or not np.all(np.isfinite(audio)):
            raise ModelInferenceError("inference audio must be one-dimensional and finite")
        return audio

    def _iter_chunks_with_timing(
        self,
        segment: SynthesisSegment,
        *,
        voice: PreparedVoice,
        generation: GenerationConfig | None = None,
    ) -> Iterator[tuple[RenderedChunk, float, float]]:
        self._ensure_open()
        if not isinstance(segment, SynthesisSegment):
            raise TypeError("segment must be a SynthesisSegment")
        if not segment.text.strip():
            raise ValueError("segment text must not be empty or whitespace")
        language = self.bundle_language
        if segment.language is not None and normalize_language(segment.language) != language:
            raise BundleLanguageError(
                f"Language {segment.language!r} is incompatible with Pocket bundle "
                f"{self.bundle_id!r} (language {language!r})"
            )
        voice.validate_compatible(bundle_id=self.bundle_id, sample_rate=self.sample_rate)
        config = generation or GenerationConfig()
        frontend_started = time.perf_counter()
        model_texts = self.frontend.split_for_model(segment.text)
        split_ms = (time.perf_counter() - frontend_started) * 1000
        if not model_texts:
            raise ValueError("segment text must not be empty or whitespace")
        for index, model_text in enumerate(model_texts):
            frontend_started = time.perf_counter()
            token_ids = self.frontend.encode(model_text)
            frontend_ms = (time.perf_counter() - frontend_started) * 1000
            if index == 0:
                frontend_ms += split_ms
            inference_started = time.perf_counter()
            audio = self.infer_tokens(token_ids, voice, config)
            inference_ms = (time.perf_counter() - inference_started) * 1000
            yield (
                RenderedChunk(
                    index=index,
                    text=model_text,
                    model_text=model_text,
                    token_ids=token_ids,
                    audio=audio,
                    sample_rate=self.sample_rate,
                ),
                frontend_ms,
                inference_ms,
            )

    def iter_chunks(
        self,
        segment: SynthesisSegment,
        *,
        voice: PreparedVoice,
        generation: GenerationConfig | None = None,
    ) -> Iterator[RenderedChunk]:
        """Yield request-local chunks split only to satisfy the Pocket token limit."""
        for chunk, _, _ in self._iter_chunks_with_timing(
            segment, voice=voice, generation=generation
        ):
            yield chunk

    def synthesize(
        self,
        segment: SynthesisSegment,
        *,
        voice: PreparedVoice,
        generation: GenerationConfig | None = None,
    ) -> RenderedSegment:
        """Render one prepared-text request as an independent Pocket result."""
        started = time.perf_counter()
        chunks: list[RenderedChunk] = []
        frontend_ms = 0.0
        inference_ms = 0.0
        for chunk, chunk_frontend_ms, chunk_inference_ms in self._iter_chunks_with_timing(
            segment, voice=voice, generation=generation
        ):
            chunks.append(chunk)
            frontend_ms += chunk_frontend_ms
            inference_ms += chunk_inference_ms
        audio = np.concatenate([chunk.audio for chunk in chunks]).astype(np.float32, copy=False)
        token_ids = tuple(token_id for chunk in chunks for token_id in chunk.token_ids)
        return RenderedSegment(
            id=segment.id,
            audio=audio,
            sample_rate=self.sample_rate,
            text=segment.text,
            language=self.bundle_language,
            token_ids=token_ids,
            chunks=tuple(chunks),
            diagnostics=self.diagnostics,
            timing=SynthesisTiming(
                frontend_ms=frontend_ms,
                inference_ms=inference_ms,
                total_ms=(time.perf_counter() - started) * 1000,
            ),
        )

    def synthesize_text(
        self,
        text: str,
        *,
        voice: PreparedVoice | Any,
        id: str = "speech",
        language: str | None = None,
        generation: GenerationConfig | None = None,
    ) -> RenderedSegment:
        """Prepare a concrete voice if needed, then synthesize one text request."""
        prepared_voice = voice if isinstance(voice, PreparedVoice) else self.prepare_voice(voice)
        return self.synthesize(
            SynthesisSegment(id=id, text=text, language=language),
            voice=prepared_voice,
            generation=generation,
        )

    @property
    def diagnostics(self) -> RuntimeDiagnostics:
        raw = runtime_diagnostics(self.runtime)
        return RuntimeDiagnostics(
            bundle_id=self.bundle_id,
            bundle_path=str(self.paths.root if self.paths is not None else "managed"),
            sample_rate=self.sample_rate,
            precision=self.precision,
            providers_requested=tuple(raw.get("providers_requested", ())),
            providers_active=tuple(raw.get("providers_active", ())),
            sessions=tuple(raw.get("sessions", ())),
            source_revision=self.source_revision,
        )

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeClosedError("PocketRuntime is closed")

    def __enter__(self) -> PocketRuntime:  # noqa: UP037
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

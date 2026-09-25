from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, replace
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
    EmptyTextError,
    InvalidGenerationConfigError,
    InvalidLanguageError,
    InvalidRequestError,
    InvalidVoiceError,
    ModelInferenceError,
    RuntimeClosedError,
    SynthesisInputTooLongError,
    UnsupportedBundleError,
    UnsupportedFeatureError,
    VoicePromptError,
)
from .frontend import PocketFrontend
from .language import bundle_language as resolve_bundle_language
from .language import normalize_language
from .types import (
    RenderedChunk,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
)
from .voice import PreparedVoice, prepare_voice
from .voice_level import VoiceLevelConfig, apply_voice_level_calibration


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
                source_revision=self.source_revision,
                predefined_voices=self.predefined_voices,
            )

        if isinstance(source, str) and source in self.predefined_voices:
            voice = _call("prepare_voice", load_voice)
        else:
            voice = load_voice()
        voice.validate_compatible(
            bundle_id=self.bundle_id,
            sample_rate=self.sample_rate,
            source_revision=self.source_revision,
        )
        if voice.bundle_id == self.bundle_id and voice.source_revision == self.source_revision:
            return voice
        return PreparedVoice(
            state=voice.state,
            sample_rate=voice.sample_rate,
            source=voice.source,
            bundle_id=self.bundle_id,
            runtime_fingerprint=self.bundle_id,
            metadata=voice.metadata,
            fingerprint=voice.fingerprint,
            source_revision=self.source_revision,
        )

    def infer_tokens(
        self,
        token_ids: Sequence[int],
        voice: PreparedVoice,
        generation: GenerationConfig,
        *,
        text_length: int | None = None,
    ) -> np.ndarray:
        self._ensure_open()
        if not isinstance(voice, PreparedVoice):
            raise InvalidVoiceError("voice must be a PreparedVoice")
        try:
            voice.validate_compatible(
                bundle_id=self.bundle_id,
                sample_rate=self.sample_rate,
                source_revision=self.source_revision,
            )
        except VoicePromptError as exc:
            raise InvalidVoiceError(str(exc)) from exc
        if not isinstance(generation, GenerationConfig):
            raise InvalidGenerationConfigError("config must be a GenerationConfig")
        if len(token_ids) > self.metadata.max_token_per_chunk:
            raise SynthesisInputTooLongError(
                text_length=text_length,
                token_count=len(token_ids),
                max_tokens=self.metadata.max_token_per_chunk,
                bundle_id=self.bundle_id,
            )
        return self._infer_tokens_unchecked(token_ids, voice, generation)

    def _infer_tokens_unchecked(
        self, token_ids: Sequence[int], voice: PreparedVoice, generation: GenerationConfig
    ) -> np.ndarray:
        self._ensure_open()
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

    def iter_chunks(
        self,
        segment: SynthesisSegment,
        *,
        voice: PreparedVoice,
        generation: GenerationConfig | None = None,
    ) -> Iterator[RenderedChunk]:
        """Yield model-limit chunks for explicitly chunked convenience rendering."""
        self._ensure_open()
        if not isinstance(segment, SynthesisSegment):
            raise InvalidRequestError("segment must be a SynthesisSegment")
        if not segment.text.strip():
            raise EmptyTextError("segment text must not be empty or whitespace")
        language = self.bundle_language
        if segment.language is not None and normalize_language(segment.language) != language:
            raise BundleLanguageError(
                f"Language {segment.language!r} is incompatible with Pocket bundle "
                f"{self.bundle_id!r} (language {language!r})"
            )
        if not isinstance(voice, PreparedVoice):
            raise InvalidVoiceError("voice must be a PreparedVoice")
        try:
            voice.validate_compatible(
                bundle_id=self.bundle_id,
                sample_rate=self.sample_rate,
                source_revision=self.source_revision,
            )
        except VoicePromptError as exc:
            raise InvalidVoiceError(str(exc)) from exc
        config = GenerationConfig() if generation is None else generation
        if not isinstance(config, GenerationConfig):
            raise InvalidGenerationConfigError("config must be a GenerationConfig")
        model_texts = self.frontend.split_for_model(segment.text)
        if not model_texts:
            raise EmptyTextError("segment text produced no Pocket model chunks")
        for index, model_text in enumerate(model_texts):
            token_ids = self.frontend.encode(model_text)
            audio = self.infer_tokens(token_ids, voice, config, text_length=len(segment.text))
            yield RenderedChunk(
                index=index,
                text=model_text,
                model_text=model_text,
                token_ids=token_ids,
                audio=audio,
                sample_rate=self.sample_rate,
            )

    def synthesize(
        self,
        request: SynthesisRequest,
        *,
        voice: PreparedVoice,
        config: GenerationConfig | None = None,
        voice_level: VoiceLevelConfig | None = None,
    ) -> SynthesisResult:
        """Render one complete request with one encoding and at most one inference."""
        self._ensure_open()
        if not isinstance(request, SynthesisRequest):
            raise InvalidRequestError("request must be a SynthesisRequest")
        if not request.text.strip():
            raise EmptyTextError("request text must not be empty or whitespace")
        if request.tokens:
            raise UnsupportedFeatureError(feature="linguistic_tokens")
        if request.pronunciation_overrides:
            raise UnsupportedFeatureError(feature="pronunciation_overrides")
        language = self.bundle_language
        if request.language is not None:
            if not request.language.strip() or normalize_language(request.language) != language:
                raise InvalidLanguageError(
                    f"Language {request.language!r} is incompatible with Pocket bundle "
                    f"{self.bundle_id!r} (language {language!r})"
                )
        generation = GenerationConfig() if config is None else config
        if not isinstance(generation, GenerationConfig):
            raise InvalidGenerationConfigError("config must be a GenerationConfig")
        voice_level_config = VoiceLevelConfig() if voice_level is None else voice_level
        if not isinstance(voice_level_config, VoiceLevelConfig):
            raise InvalidGenerationConfigError("voice_level must be a VoiceLevelConfig")
        if not isinstance(voice, PreparedVoice):
            raise InvalidVoiceError("voice must be a PreparedVoice")
        try:
            voice.validate_compatible(
                bundle_id=self.bundle_id,
                sample_rate=self.sample_rate,
                source_revision=self.source_revision,
            )
        except VoicePromptError as exc:
            raise InvalidVoiceError(str(exc)) from exc

        started = time.perf_counter()
        frontend_started = time.perf_counter()
        token_ids = self.frontend.encode(request.text)
        frontend_ms = (time.perf_counter() - frontend_started) * 1000
        if not token_ids:
            raise EmptyTextError("request text produced no Pocket tokens")
        max_tokens = self.metadata.max_token_per_chunk
        if len(token_ids) > max_tokens:
            raise SynthesisInputTooLongError(
                text_length=len(request.text),
                token_count=len(token_ids),
                max_tokens=max_tokens,
                bundle_id=self.bundle_id,
            )

        inference_started = time.perf_counter()
        audio = self._infer_tokens_unchecked(token_ids, voice, generation)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        voice_identity = voice.identity
        calibration_catalog: Mapping[str, object] = {}
        if voice_level_config.mode == "calibrated" and voice_level_config.gain_db is None:
            raw_metadata = self.metadata.raw or {}
            raw_catalog = raw_metadata.get("voice_level_calibration", {})
            if not isinstance(raw_catalog, Mapping):
                raise UnsupportedBundleError("bundle voice_level_calibration must be an object")
            calibration_catalog = raw_catalog
        audio, voice_level_application = apply_voice_level_calibration(
            audio,
            voice_level_config,
            identity=voice_identity,
            catalog=calibration_catalog,
        )
        total_ms = (time.perf_counter() - started) * 1000
        diagnostics = self.diagnostics
        return SynthesisResult(
            id=request.id,
            audio=audio,
            sample_rate=self.sample_rate,
            text=request.text,
            language=language,
            metadata={
                "bundle_id": self.bundle_id,
                "bundle_revision": self.source_revision,
                "voice_identity": voice_identity,
                "token_count": len(token_ids),
                "generation_config": asdict(generation),
                "voice_level_config": asdict(voice_level_config),
                "voice_level_application": asdict(voice_level_application),
                "runtime_diagnostics": asdict(diagnostics),
                "timing": asdict(
                    SynthesisTiming(
                        frontend_ms=frontend_ms,
                        inference_ms=inference_ms,
                        total_ms=total_ms,
                    )
                ),
            },
        )

    def synthesize_text(
        self,
        text: str,
        *,
        voice: PreparedVoice | Any,
        id: str = "speech",
        language: str | None = None,
        config: GenerationConfig | None = None,
        voice_level: VoiceLevelConfig | None = None,
    ) -> SynthesisResult:
        """Prepare a voice for one strict, unsplit text request."""
        request = SynthesisRequest(id=id, text=text, language=language)
        if not request.text.strip():
            raise EmptyTextError("request text must not be empty or whitespace")
        try:
            prepared_voice = (
                voice if isinstance(voice, PreparedVoice) else self.prepare_voice(voice)
            )
        except VoicePromptError as exc:
            raise InvalidVoiceError(str(exc)) from exc
        return self.synthesize(
            request,
            voice=prepared_voice,
            config=config,
            voice_level=voice_level,
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

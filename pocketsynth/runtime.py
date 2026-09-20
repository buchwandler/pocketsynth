from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._onnxvoice import (
    ResolvedPocketBundle,
    open_installed_bundle,
    open_local_bundle,
    runtime_diagnostics,
)
from .bundle import BundleMetadata, BundlePaths, Precision
from .config import GenerationConfig
from .diagnostics import RuntimeDiagnostics
from .errors import ModelInferenceError, PipelineClosedError
from .frontend import PocketFrontend
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
    ) -> "PocketRuntime":  # noqa: UP037
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
    def from_resolved(
        cls,
        resolved: ResolvedPocketBundle,
        *,
        providers: Sequence[Any] | str | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
    ) -> "PocketRuntime":  # noqa: UP037
        metadata = BundleMetadata.load(resolved.metadata_path)
        runtime = open_installed_bundle(
            resolved,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
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
    def sample_rate(self) -> int:
        return self.metadata.sample_rate

    def prepare_voice(self, source: Any) -> PreparedVoice:
        self._ensure_open()
        voice = prepare_voice(self.runtime, source, sample_rate=self.sample_rate)
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
        return np.asarray(result.audio, dtype=np.float32)

    @property
    def diagnostics(self) -> RuntimeDiagnostics:
        raw = runtime_diagnostics(self.runtime)
        return RuntimeDiagnostics(
            bundle_id=self.bundle_id,
            bundle_path=str(self.paths.root if self.paths is not None else "managed"),
            sample_rate=self.sample_rate,
            precision=self.precision,
            providers_requested=tuple(raw.get("providers_requested", ())),
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
            raise PipelineClosedError("PocketRuntime is closed")

    def __enter__(self) -> "PocketRuntime":  # noqa: UP037
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

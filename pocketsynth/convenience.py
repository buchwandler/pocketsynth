"""Convenience functions for quick PocketSynth usage."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from utterplan import LinguisticsConfig, PauseConfig, SSMDConfig

from .audio import write_wav
from .config import GenerationConfig
from .pipeline import PocketPipeline
from .types import AudioResult


def _save_wav_atomically(result: AudioResult, destination: Path) -> None:
    """Write a WAV file atomically, creating parent directories."""
    if destination.exists() and destination.is_dir():
        raise ValueError(f"output path is a directory: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        write_wav(temporary, result.audio, result.sample_rate)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def synthesize(
    text: str,
    *,
    bundle: str,
    voice: Any,
    precision: Literal["int8", "fp32"] = "int8",
    temperature: float = 0.7,
    lsd_steps: int = 1,
    max_frames: int | None = None,
    frames_after_eos: int | None = None,
    normalize_audio: bool = False,
    volume: float = 1.0,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh_catalog: bool = False,
    force_download: bool = False,
    document_format: Literal["plain", "ssmd"] = "plain",
    text_preparation: Literal["identity", "spokenform"] = "spokenform",
    language: str | None = None,
    unit: Literal["paragraph", "sentence"] = "sentence",
    pauses: PauseConfig | None = None,
    linguistics: LinguisticsConfig | None = None,
    ssmd: SSMDConfig | None = None,
    progress: Any | None = None,
) -> AudioResult:
    """Synthesize text to an AudioResult using a managed bundle.

    Args:
        text: Text to synthesize.
        bundle: Managed bundle name (e.g. "english_2026-04").
        voice: Bundle-declared predefined name, reference WAV path, in-memory audio tuple,
        or PreparedVoice.
        precision: Model precision ("int8" or "fp32").
        temperature: Sampling temperature.
        lsd_steps: LSD steps.
        max_frames: Maximum frames to generate.
        frames_after_eos: Frames after EOS token.
        normalize_audio: Normalize audio amplitude.
        volume: Volume scaling factor.
        providers: ONNX Runtime providers.
        provider_options: Provider options.
        session_options: Session options.
        cache_dir: Cache directory for managed bundles.
        offline: Disable network access.
        refresh_catalog: Force catalog refresh.
        force_download: Force re-download.
        document_format: Document format ("plain" or "ssmd").
        text_preparation: Text preparation mode.
        language: Language override.
        unit: Planning unit ("paragraph" or "sentence").
        pauses: Pause configuration.
        linguistics: Linguistics configuration.
        ssmd: SSMD configuration.
        progress: Progress callback.

    Returns:
        AudioResult with synthesized audio.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(bundle, str) or not bundle:
        raise ValueError("bundle must be a non-empty string")

    generation = GenerationConfig(
        temperature=temperature,
        lsd_steps=lsd_steps,
        max_frames=max_frames,
        frames_after_eos=frames_after_eos,
        normalize_audio=normalize_audio,
        volume=volume,
    )

    with PocketPipeline.from_pretrained(
        bundle,
        precision=precision,
        cache_dir=cache_dir,
        offline=offline,
        refresh_catalog=refresh_catalog,
        force_download=force_download,
        generation=generation,
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
        document_format=document_format,
        text_preparation=text_preparation,
        language=language,
        unit=unit,
        pauses=pauses,
        linguistics=linguistics,
        ssmd=ssmd,
        progress=progress,
    ) as pipeline:
        pipeline.set_default_voice(voice)
        return pipeline(text)


def synthesize_to_wav(
    text: str,
    output: str | Path,
    *,
    bundle: str,
    voice: Any,
    precision: Literal["int8", "fp32"] = "int8",
    temperature: float = 0.7,
    lsd_steps: int = 1,
    max_frames: int | None = None,
    frames_after_eos: int | None = None,
    normalize_audio: bool = False,
    volume: float = 1.0,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh_catalog: bool = False,
    force_download: bool = False,
    document_format: Literal["plain", "ssmd"] = "plain",
    text_preparation: Literal["identity", "spokenform"] = "spokenform",
    language: str | None = None,
    unit: Literal["paragraph", "sentence"] = "sentence",
    pauses: PauseConfig | None = None,
    linguistics: LinguisticsConfig | None = None,
    ssmd: SSMDConfig | None = None,
    progress: Any | None = None,
) -> Path:
    """Synthesize text to a WAV file using a managed bundle.

    Creates parent directories and writes atomically.

    Args:
        text: Text to synthesize.
        output: Output WAV file path.
        bundle: Managed bundle name (e.g. "english_2026-04").
        voice: Bundle-declared predefined name, reference WAV path, in-memory audio tuple,
        or PreparedVoice.
        precision: Model precision ("int8" or "fp32").
        temperature: Sampling temperature.
        lsd_steps: LSD steps.
        max_frames: Maximum frames to generate.
        frames_after_eos: Frames after EOS token.
        normalize_audio: Normalize audio amplitude.
        volume: Volume scaling factor.
        providers: ONNX Runtime providers.
        provider_options: Provider options.
        session_options: Session options.
        cache_dir: Cache directory for managed bundles.
        offline: Disable network access.
        refresh_catalog: Force catalog refresh.
        force_download: Force re-download.
        document_format: Document format ("plain" or "ssmd").
        text_preparation: Text preparation mode.
        language: Language override.
        unit: Planning unit ("paragraph" or "sentence").
        pauses: Pause configuration.
        linguistics: Linguistics configuration.
        ssmd: SSMD configuration.
        progress: Progress callback.

    Returns:
        Path to the written WAV file.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(bundle, str) or not bundle:
        raise ValueError("bundle must be a non-empty string")

    destination = Path(output)
    generation = GenerationConfig(
        temperature=temperature,
        lsd_steps=lsd_steps,
        max_frames=max_frames,
        frames_after_eos=frames_after_eos,
        normalize_audio=normalize_audio,
        volume=volume,
    )

    with PocketPipeline.from_pretrained(
        bundle,
        precision=precision,
        cache_dir=cache_dir,
        offline=offline,
        refresh_catalog=refresh_catalog,
        force_download=force_download,
        generation=generation,
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
        document_format=document_format,
        text_preparation=text_preparation,
        language=language,
        unit=unit,
        pauses=pauses,
        linguistics=linguistics,
        ssmd=ssmd,
        progress=progress,
    ) as pipeline:
        pipeline.set_default_voice(voice)
        result = pipeline(text)
        _save_wav_atomically(result, destination)

    return destination

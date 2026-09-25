"""Convenience functions for standalone PocketSynth use."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .asset_progress import AssetProgressCallback
from .audio import write_wav
from .config import GenerationConfig
from .runtime import PocketRuntime
from .text_split import SentenceSplitMode
from .types import RenderedSegment


def _save_wav_atomically(result: RenderedSegment, destination: Path) -> None:
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
        with temporary.open("rb+") as handle:
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
    language: str | None = None,
    sentence_split: SentenceSplitMode = "phrasplit",
    temperature: float = 0.7,
    lsd_steps: int = 1,
    max_frames: int | None = None,
    frames_after_eos: int | None = None,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh_catalog: bool = False,
    force_download: bool = False,
    progress: AssetProgressCallback | None = None,
) -> RenderedSegment:
    """Synthesize already-prepared speakable text with a managed Pocket bundle."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(bundle, str) or not bundle:
        raise ValueError("bundle must be a non-empty string")

    generation = GenerationConfig(
        temperature=temperature,
        lsd_steps=lsd_steps,
        max_frames=max_frames,
        frames_after_eos=frames_after_eos,
    )
    with PocketRuntime.from_pretrained(
        bundle,
        precision=precision,
        cache_dir=cache_dir,
        offline=offline,
        refresh_catalog=refresh_catalog,
        force_download=force_download,
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
        progress=progress,
    ) as runtime:
        prepared_voice = runtime.prepare_voice(voice)
        return runtime.synthesize_text(
            text,
            voice=prepared_voice,
            language=language,
            generation=generation,
            sentence_split=sentence_split,
        )


def synthesize_to_wav(
    text: str,
    output: str | Path,
    *,
    bundle: str,
    voice: Any,
    precision: Literal["int8", "fp32"] = "int8",
    language: str | None = None,
    sentence_split: SentenceSplitMode = "phrasplit",
    temperature: float = 0.7,
    lsd_steps: int = 1,
    max_frames: int | None = None,
    frames_after_eos: int | None = None,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh_catalog: bool = False,
    force_download: bool = False,
    progress: AssetProgressCallback | None = None,
) -> Path:
    """Synthesize text to an atomically written mono PCM16 WAV file."""
    destination = Path(output)
    result = synthesize(
        text,
        bundle=bundle,
        voice=voice,
        precision=precision,
        language=language,
        temperature=temperature,
        lsd_steps=lsd_steps,
        max_frames=max_frames,
        frames_after_eos=frames_after_eos,
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
        cache_dir=cache_dir,
        offline=offline,
        refresh_catalog=refresh_catalog,
        force_download=force_download,
        sentence_split=sentence_split,
        progress=progress,
    )
    _save_wav_atomically(result, destination)
    return destination

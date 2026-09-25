"""Convenience functions for standalone PocketSynth use."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .asset_progress import AssetProgressCallback
from .audio import write_wav
from .config import GenerationConfig
from .runtime import PocketRuntime
from .text_split import SentenceSplitMode, split_text_for_synthesis
from .types import RenderedChunk, RenderedSegment, SynthesisSegment


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


def synthesize_with_runtime(
    runtime: PocketRuntime,
    text: str,
    *,
    voice: Any,
    sentence_split: SentenceSplitMode = "none",
    generation: GenerationConfig | None = None,
    language: str | None = None,
    id: str = "speech",
) -> RenderedSegment:
    """Render text through explicit sentence and model-limit convenience splitting."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty or whitespace")
    split_texts = split_text_for_synthesis(
        text, language=runtime.bundle_language, mode=sentence_split
    )
    if not split_texts:
        raise ValueError("text produced no sentence segments")
    prepared_voice = runtime.prepare_voice(voice)
    chunks: list[RenderedChunk] = []
    for split_text in split_texts:
        segment = SynthesisSegment(id=id, text=split_text, language=language)
        for chunk in runtime.iter_chunks(segment, voice=prepared_voice, generation=generation):
            chunks.append(
                RenderedChunk(
                    index=len(chunks),
                    text=chunk.text,
                    model_text=chunk.model_text,
                    token_ids=chunk.token_ids,
                    audio=chunk.audio,
                    sample_rate=chunk.sample_rate,
                    metadata=chunk.metadata,
                )
            )
    if not chunks:
        raise ValueError("text produced no Pocket model chunks")
    audio = np.concatenate([chunk.audio for chunk in chunks]).astype(np.float32, copy=False)
    token_ids = tuple(token_id for chunk in chunks for token_id in chunk.token_ids)
    return RenderedSegment(
        id=id,
        audio=audio,
        sample_rate=runtime.sample_rate,
        text=text,
        language=runtime.bundle_language,
        token_ids=token_ids,
        chunks=tuple(chunks),
        diagnostics=runtime.diagnostics,
        metadata={"sentence_split": sentence_split},
    )


def synthesize(
    text: str,
    *,
    bundle: str,
    voice: Any,
    precision: Literal["int8", "fp32"] = "int8",
    language: str | None = None,
    sentence_split: SentenceSplitMode = "none",
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
        return synthesize_with_runtime(
            runtime,
            text,
            voice=voice,
            sentence_split=sentence_split,
            generation=generation,
            language=language,
        )


def synthesize_to_wav(
    text: str,
    output: str | Path,
    *,
    bundle: str,
    voice: Any,
    precision: Literal["int8", "fp32"] = "int8",
    language: str | None = None,
    sentence_split: SentenceSplitMode = "none",
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

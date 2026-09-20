from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import as_float32_mono
from .errors import VoicePromptError


@dataclass(frozen=True, slots=True)
class PreparedVoice:
    state: Any
    sample_rate: int
    source: str | None = None
    bundle_id: str | None = None
    runtime_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_compatible(self, *, bundle_id: str | None = None, sample_rate: int | None = None) -> None:
        """Check that this voice is compatible with the target runtime."""
        if sample_rate is not None and self.sample_rate != sample_rate:
            raise VoicePromptError(
                f"PreparedVoice sample rate {self.sample_rate} does not match target {sample_rate}"
            )
        if bundle_id is not None and self.bundle_id is not None and self.bundle_id != bundle_id:
            raise VoicePromptError(
                f"PreparedVoice bundle {self.bundle_id!r} does not match target {bundle_id!r}"
            )

def _read_pcm_wav(path: str | Path) -> tuple[np.ndarray, int]:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            compression = handle.getcomptype()
            frames = handle.readframes(handle.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise VoicePromptError(
            f"Expected mono PCM WAV voice prompt; could not read {source}: {exc}"
        ) from exc
    if channels != 1 or width != 2 or compression != "NONE" or sample_rate <= 0:
        raise VoicePromptError(
            "Expected mono PCM WAV voice prompt; "
            f"actual channels={channels}, sample width={width} bytes, "
            f"sample rate={sample_rate} Hz, compression={compression!r}"
        )
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate

def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    audio = as_float32_mono(audio)
    if source_rate == target_rate or not audio.size:
        return audio
    target_length = max(1, round(audio.size * target_rate / source_rate))
    old_x = np.linspace(0.0, 1.0, audio.size, endpoint=True)
    new_x = np.linspace(0.0, 1.0, target_length, endpoint=True)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def prepare_voice(runtime: Any, source: str | Path | tuple[np.ndarray, int] | PreparedVoice, *, sample_rate: int) -> PreparedVoice:
    if isinstance(source, PreparedVoice):
        return source
    if isinstance(source, tuple):
        audio, source_rate = source
        label = None
    else:
        audio, source_rate = _read_pcm_wav(source)
        label = str(source)
    audio = _resample_linear(audio, int(source_rate), sample_rate)
    method = getattr(runtime, "prepare_voice", None)
    if method is None:
        raise VoicePromptError(
            "OnnxVoice PocketAdapter must provide prepare_voice(audio, sample_rate=...)"
        )
    state = method(audio, sample_rate=sample_rate)
    return PreparedVoice(state=state, sample_rate=sample_rate, source=label)

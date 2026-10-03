from __future__ import annotations

import hashlib
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._onnxvoice import ResolvedVoicePrompt
from .audio import as_float32_mono
from .errors import VoicePromptError
from .voice_prompts import VoicePromptInfo


@dataclass(frozen=True, slots=True)
class PreparedVoice:
    state: Any
    sample_rate: int
    source: str | None = None
    bundle_id: str | None = None
    runtime_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprint: str | None = None
    source_revision: str | None = None
    voice_prompt: VoicePromptInfo | None = None

    @property
    def bundle_revision(self) -> str | None:
        """The active Pocket bundle revision; prompt revisions live on voice_prompt."""
        return self.source_revision

    @property
    def identity(self) -> dict[str, str | int | None] | None:
        kind = self.metadata.get("kind")
        if kind == "predefined":
            name = self.metadata.get("name")
            if not isinstance(name, str) or not name:
                return None
            return {
                "kind": "predefined",
                "bundle_id": self.bundle_id,
                "name": name,
                "source_revision": self.source_revision,
            }
        if self.voice_prompt is not None and self.fingerprint is not None:
            return {
                "kind": "managed_reference",
                "ref": self.voice_prompt.ref,
                "source_sha256": self.voice_prompt.sha256,
                "source_revision": self.voice_prompt.source_revision,
                "prepared_sha256": self.fingerprint,
                "sample_rate": self.sample_rate,
            }
        if kind == "reference" and self.fingerprint is not None:
            return {
                "kind": "reference",
                "sha256": self.fingerprint,
                "sample_rate": self.sample_rate,
            }
        return None

    def validate_compatible(
        self,
        *,
        bundle_id: str | None = None,
        sample_rate: int | None = None,
        source_revision: str | None = None,
    ) -> None:
        """Check that this voice is compatible with the target runtime."""
        if sample_rate is not None and self.sample_rate != sample_rate:
            raise VoicePromptError(
                f"PreparedVoice sample rate {self.sample_rate} does not match target {sample_rate}"
            )
        if bundle_id is not None and self.bundle_id is not None and self.bundle_id != bundle_id:
            raise VoicePromptError(
                f"PreparedVoice bundle {self.bundle_id!r} does not match target {bundle_id!r}"
            )
        if (
            source_revision is not None
            and self.source_revision is not None
            and self.source_revision != source_revision
        ):
            raise VoicePromptError(
                f"PreparedVoice source revision {self.source_revision!r} does not match "
                f"target {source_revision!r}"
            )


def _read_pcm_wav(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        source = Path(path)
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            compression = handle.getcomptype()
            frames = handle.readframes(handle.getnframes())
    except (OSError, EOFError, TypeError, ValueError, wave.Error) as exc:
        raise VoicePromptError(
            f"Expected uncompressed PCM WAV voice prompt; could not read {path!s}: {exc}"
        ) from exc
    if channels < 1 or width not in (1, 2, 3, 4) or compression != "NONE" or sample_rate <= 0:
        raise VoicePromptError(
            "Expected uncompressed PCM WAV voice prompt with 1-, 2-, 3-, or 4-byte samples; "
            f"actual channels={channels}, sample width={width} bytes, "
            f"sample rate={sample_rate} Hz, compression={compression!r}"
        )
    if not frames:
        raise VoicePromptError("WAV voice prompt contains no audio frames")
    sample_count = len(frames) // width
    if sample_count * width != len(frames) or sample_count % channels:
        raise VoicePromptError("WAV voice prompt contains an incomplete PCM frame")
    if width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        packed = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        signed = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        signed[signed >= 0x800000] -= 0x1000000
        samples = signed.astype(np.float32) / 8388608.0
    else:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    channels_audio = samples.reshape(-1, channels)
    audio = channels_audio.mean(axis=1, dtype=np.float32) if channels > 1 else channels_audio[:, 0]
    return as_float32_mono(audio), sample_rate


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    audio = as_float32_mono(audio)
    if source_rate == target_rate or not audio.size:
        return audio
    target_length = max(1, round(audio.size * target_rate / source_rate))
    old_x = np.linspace(0.0, 1.0, audio.size, endpoint=True)
    new_x = np.linspace(0.0, 1.0, target_length, endpoint=True)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def load_reference_audio(path: str | Path, *, target_sample_rate: int) -> np.ndarray:
    audio, source_rate = _read_pcm_wav(path)
    return _resample_linear(audio, source_rate, target_sample_rate)


def _looks_like_path_string(value: str) -> bool:
    path = Path(value)
    return bool(path.suffix) or "/" in value or "\\" in value or path.exists()


def _is_managed_voice_prompt_ref(value: str) -> bool:
    return value.startswith("kyutai-tts-voices:")


def _reference_fingerprint(audio: np.ndarray, sample_rate: int) -> str:
    digest = hashlib.sha256(b"pocketsynth:reference-voice-v1\0")
    digest.update(sample_rate.to_bytes(8, "big", signed=False))
    digest.update(np.asarray(audio, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def _predefined_fingerprint(bundle_id: str | None, name: str, source_revision: str | None) -> str:
    digest = hashlib.sha256(b"pocketsynth:predefined-voice-v2\0")
    for value in (bundle_id or "", name, source_revision or ""):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def prepare_voice(
    runtime: Any,
    source: str | Path | tuple[np.ndarray, int] | PreparedVoice | VoicePromptInfo,
    *,
    sample_rate: int,
    bundle_id: str | None = None,
    source_revision: str | None = None,
    predefined_voices: tuple[str, ...] = (),
    managed_voice_resolver: Callable[[str | VoicePromptInfo], ResolvedVoicePrompt] | None = None,
) -> PreparedVoice:
    if isinstance(source, PreparedVoice):
        return source
    metadata: dict[str, Any] = {"kind": "reference"}
    managed_info: VoicePromptInfo | None = None
    if isinstance(source, tuple):
        audio, source_rate = source
        label = None
    elif isinstance(source, str) and source in predefined_voices:
        method = getattr(runtime, "prepare_predefined_voice", None)
        if method is None:
            raise VoicePromptError(
                "OnnxVoice PocketAdapter must provide prepare_predefined_voice(name)"
            )
        state = method(source)
        return PreparedVoice(
            state=state,
            sample_rate=sample_rate,
            source=source,
            bundle_id=bundle_id,
            runtime_fingerprint=bundle_id,
            metadata={
                "kind": "predefined",
                "name": source,
                "source_revision": source_revision,
            },
            fingerprint=_predefined_fingerprint(bundle_id, source, source_revision),
            source_revision=source_revision,
        )
    elif isinstance(source, VoicePromptInfo) or (
        isinstance(source, str) and _is_managed_voice_prompt_ref(source)
    ):
        if managed_voice_resolver is None:
            raise VoicePromptError("Managed voice prompts require a configured OnnxVoice resolver")
        prompt = managed_voice_resolver(source)
        managed_info = VoicePromptInfo(
            ref=prompt.ref,
            source_repository=prompt.source_repository,
            source_revision=prompt.source_revision,
            source_path=prompt.source_path,
            size=prompt.size,
            sha256=prompt.sha256,
            license=prompt.license,
            dataset=prompt.dataset,
            variant=prompt.variant,
        )
        audio = load_reference_audio(prompt.path, target_sample_rate=sample_rate)
        source_rate = sample_rate
        label = managed_info.ref
        metadata.update(
            {
                "kind": "managed_reference",
                "managed_ref": managed_info.ref,
                "source_repository": managed_info.source_repository,
                "source_revision": managed_info.source_revision,
                "source_path": managed_info.source_path,
                "source_sha256": managed_info.sha256,
                "license": managed_info.license,
                "dataset": managed_info.dataset,
                "variant": managed_info.variant,
            }
        )
    elif isinstance(source, str) and not _looks_like_path_string(source):
        names = ", ".join(predefined_voices) or "none"
        raise VoicePromptError(
            f"Unknown predefined Pocket voice {source!r}. Available voices: {names}."
        )
    else:
        audio = load_reference_audio(source, target_sample_rate=sample_rate)
        source_rate = sample_rate
        label = str(source)
    audio = _resample_linear(audio, int(source_rate), sample_rate)
    method = getattr(runtime, "prepare_voice", None)
    if method is None:
        raise VoicePromptError(
            "OnnxVoice PocketAdapter must provide prepare_voice(audio, sample_rate=...)"
        )
    state = method(audio, sample_rate=sample_rate)
    return PreparedVoice(
        state=state,
        sample_rate=sample_rate,
        source=label,
        bundle_id=bundle_id,
        runtime_fingerprint=bundle_id,
        metadata=metadata,
        fingerprint=_reference_fingerprint(audio, sample_rate),
        source_revision=source_revision,
        voice_prompt=managed_info,
    )

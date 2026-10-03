import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pocketsynth._onnxvoice import ResolvedVoicePrompt
from pocketsynth.errors import VoicePromptError
from pocketsynth.voice import (
    PreparedVoice,
    _read_pcm_wav,
    _resample_linear,
    load_reference_audio,
    prepare_voice,
)
from pocketsynth.voice_prompts import VoicePromptInfo


def test_read_wav_and_resample(tmp_path):
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.array([0, 1000, -1000, 0], dtype="<i2").tobytes())
    audio, rate = _read_pcm_wav(path)
    assert rate == 16000
    assert audio.dtype == np.float32
    resampled = _resample_linear(audio, 16000, 24000)
    assert resampled.ndim == 1
    assert len(resampled) == 6
    normalized = load_reference_audio(path, target_sample_rate=24000)
    np.testing.assert_array_equal(normalized, resampled)


def _write_wav(path: Path, *, channels: int, width: int, frames: bytes, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(frames)


@pytest.mark.parametrize(
    ("width", "frames", "expected"),
    [
        (1, bytes([0, 128, 255]), [-1.0, 0.0, 127.0 / 128.0]),
        (2, np.array([-32768, 0, 32767], dtype="<i2").tobytes(), [-1.0, 0.0, 32767 / 32768]),
        (
            3,
            b"".join((value & 0xFFFFFF).to_bytes(3, "little") for value in (-8388608, 0, 8388607)),
            [-1.0, 0.0, 8388607 / 8388608],
        ),
        (
            4,
            np.array([-2147483648, 0, 2147483647], dtype="<i4").tobytes(),
            [-1.0, 0.0, 1.0],
        ),
    ],
)
def test_read_wav_normalizes_supported_pcm_widths(
    tmp_path: Path, width: int, frames: bytes, expected: list[float]
) -> None:
    path = tmp_path / f"pcm{width * 8}.wav"
    _write_wav(path, channels=1, width=width, frames=frames)

    audio, sample_rate = _read_pcm_wav(path)

    assert sample_rate == 16_000
    assert audio.dtype == np.float32
    assert np.all(np.isfinite(audio))
    np.testing.assert_allclose(audio, expected, atol=1e-6)


def test_read_stereo_wav_downmixes_to_mono(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    frames = np.array([32767, 0, 0, -32768], dtype="<i2").tobytes()
    _write_wav(path, channels=2, width=2, frames=frames)

    audio, sample_rate = _read_pcm_wav(path)

    assert sample_rate == 16_000
    assert audio.shape == (2,)
    np.testing.assert_allclose(audio, [32767 / 65536, -0.5], atol=1e-6)


def test_read_empty_wav_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    _write_wav(path, channels=1, width=2, frames=b"")

    with pytest.raises(VoicePromptError, match="no audio frames"):
        _read_pcm_wav(path)


@pytest.mark.parametrize(
    ("channels", "width", "compression"),
    [(1, 5, "NONE"), (1, 2, "ULAW")],
)
def test_read_wav_reports_unsupported_format(
    tmp_path: Path, channels: int, width: int, compression: str
) -> None:
    reader = MagicMock()
    reader.getnchannels.return_value = channels
    reader.getsampwidth.return_value = width
    reader.getframerate.return_value = 16_000
    reader.getcomptype.return_value = compression
    reader.getnframes.return_value = 1
    reader.readframes.return_value = b"\\x00" * max(width, 1)
    stream = MagicMock()
    stream.__enter__.return_value = reader

    with (
        patch("pocketsynth.voice.wave.open", return_value=stream),
        pytest.raises(
            VoicePromptError, match="Expected uncompressed PCM WAV voice prompt"
        ) as caught,
    ):
        _read_pcm_wav(tmp_path / "invalid.wav")

    assert f"channels={channels}" in str(caught.value)
    assert f"sample width={width} bytes" in str(caught.value)
    assert "sample rate=16000 Hz" in str(caught.value)


def test_predefined_name_uses_runtime_preparation_without_wav_read() -> None:
    runtime = MagicMock()
    state = object()
    runtime.prepare_predefined_voice.return_value = state

    with patch("pocketsynth.voice._read_pcm_wav") as read_wav:
        voice = prepare_voice(
            runtime,
            "alba",
            sample_rate=24000,
            bundle_id="english_2026-04",
            predefined_voices=("alba",),
        )

    read_wav.assert_not_called()
    runtime.prepare_predefined_voice.assert_called_once_with("alba")
    runtime.prepare_voice.assert_not_called()
    assert voice.state is state
    assert voice.bundle_id == "english_2026-04"
    assert voice.runtime_fingerprint == "english_2026-04"
    assert voice.fingerprint is not None
    assert voice.metadata == {
        "kind": "predefined",
        "name": "alba",
        "source_revision": None,
    }
    assert voice.identity == {
        "kind": "predefined",
        "bundle_id": "english_2026-04",
        "name": "alba",
        "source_revision": None,
    }


def test_unknown_bare_name_reports_bundle_voices_before_wav_read() -> None:
    runtime = MagicMock()
    with (
        patch("pocketsynth.voice._read_pcm_wav") as read_wav,
        pytest.raises(VoicePromptError, match="Unknown predefined.*Available voices: alba"),
    ):
        prepare_voice(
            runtime,
            "unknown",
            sample_rate=24000,
            predefined_voices=("alba",),
        )
    read_wav.assert_not_called()
    runtime.prepare_predefined_voice.assert_not_called()


def test_path_object_matching_voice_name_remains_a_local_prompt() -> None:
    runtime = MagicMock()
    audio = np.ones(4, dtype=np.float32)
    source = Path("alba")
    with patch("pocketsynth.voice._read_pcm_wav", return_value=(audio, 24000)) as read_wav:
        voice = prepare_voice(
            runtime,
            source,
            sample_rate=24000,
            bundle_id="english_2026-04",
            predefined_voices=("alba",),
        )

    read_wav.assert_called_once_with(source)
    runtime.prepare_predefined_voice.assert_not_called()
    runtime.prepare_voice.assert_called_once()
    assert voice.source == str(source)


def test_local_file_string_remains_a_reference_prompt(tmp_path: Path) -> None:
    runtime = MagicMock()
    audio = np.ones(4, dtype=np.float32)
    source = str(tmp_path / "reference.wav")
    with patch("pocketsynth.voice._read_pcm_wav", return_value=(audio, 24000)) as read_wav:
        voice = prepare_voice(
            runtime,
            source,
            sample_rate=24000,
            predefined_voices=("alba",),
        )

    read_wav.assert_called_once_with(source)
    runtime.prepare_predefined_voice.assert_not_called()
    assert voice.source == source


def test_in_memory_audio_and_prepared_voice_paths_are_preserved() -> None:
    runtime = MagicMock()
    audio = np.ones(4, dtype=np.float32)
    prepared = PreparedVoice(state=object(), sample_rate=24000)

    assert (
        prepare_voice(
            runtime,
            (audio, 24000),
            sample_rate=24000,
            predefined_voices=("alba",),
        ).sample_rate
        == 24000
    )
    assert (
        prepare_voice(
            runtime,
            prepared,
            sample_rate=24000,
            predefined_voices=("alba",),
        )
        is prepared
    )
    runtime.prepare_predefined_voice.assert_not_called()
    runtime.prepare_voice.assert_called_once()


def test_invalid_pathlike_is_reported_as_voice_prompt_error() -> None:
    class InvalidPath:
        def __fspath__(self) -> str:
            raise TypeError("invalid path")

    with pytest.raises(VoicePromptError, match="could not read"):
        prepare_voice(
            MagicMock(),
            InvalidPath(),
            sample_rate=24000,
            predefined_voices=("alba",),
        )


def test_reference_voice_fingerprint_uses_canonical_audio(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    alias = tmp_path / "alias.wav"
    different = tmp_path / "different.wav"
    samples = np.array([0, 1000, -1000, 250], dtype="<i2")

    def write_prompt(path: Path, audio: np.ndarray) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24_000)
            handle.writeframes(audio.tobytes())

    write_prompt(reference, samples)
    alias.symlink_to(reference)
    write_prompt(different, samples + 1)

    def prepare(path: Path) -> PreparedVoice:
        return prepare_voice(
            MagicMock(),
            str(path),
            sample_rate=24_000,
            bundle_id="english_2026-04",
        )

    first = prepare(reference)
    same_audio = prepare(alias)
    other_audio = prepare(different)

    assert first.fingerprint == same_audio.fingerprint
    assert first.fingerprint != other_audio.fingerprint
    assert first.identity == same_audio.identity
    assert first.identity == {
        "kind": "reference",
        "sha256": first.fingerprint,
        "sample_rate": 24_000,
    }
    assert first.identity != other_audio.identity


def test_predefined_voice_fingerprint_includes_bundle_name_and_revision() -> None:
    def prepare(bundle_id: str, name: str, source_revision: str | None = None) -> PreparedVoice:
        runtime = MagicMock()
        return prepare_voice(
            runtime,
            name,
            sample_rate=24_000,
            bundle_id=bundle_id,
            source_revision=source_revision,
            predefined_voices=(name,),
        )

    first = prepare("english_2026-04", "alba", "revision-1")
    same = prepare("english_2026-04", "alba", "revision-1")
    assert first.fingerprint == same.fingerprint
    assert first.identity == same.identity
    assert first.identity == {
        "kind": "predefined",
        "bundle_id": "english_2026-04",
        "name": "alba",
        "source_revision": "revision-1",
    }
    assert first.fingerprint != prepare("english_2026-04", "alba", "revision-2").fingerprint
    assert first.fingerprint != prepare("french_24l", "alba", "revision-1").fingerprint
    assert first.fingerprint != prepare("english_2026-04", "voice2", "revision-1").fingerprint


def test_managed_voice_ref_resolves_before_path_detection_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    ref = "kyutai-tts-voices:alba-mackenna/casual"
    prompt = ResolvedVoicePrompt(
        ref=ref,
        path=tmp_path / "casual.wav",
        source_repository="kyutai/tts-voices",
        source_revision="pinned-revision",
        source_path="alba-mackenna/casual.wav",
        size=1234,
        sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        license="cc-by-4.0",
        dataset="alba-mackenna",
        variant="casual",
    )
    audio = np.array([0.25, -0.5], dtype=np.float32)
    runtime = MagicMock()
    resolver = MagicMock(return_value=prompt)

    with (
        patch("pocketsynth.voice._looks_like_path_string") as path_check,
        patch("pocketsynth.voice._read_pcm_wav", return_value=(audio, 24_000)) as read_wav,
    ):
        voice = prepare_voice(
            runtime,
            ref,
            sample_rate=24_000,
            bundle_id="english_2026-04",
            managed_voice_resolver=resolver,
        )

    resolver.assert_called_once_with(ref)
    read_wav.assert_called_once_with(prompt.path)
    path_check.assert_not_called()
    runtime.prepare_voice.assert_called_once_with(audio, sample_rate=24_000)
    assert voice.source == ref
    assert voice.metadata == {
        "kind": "managed_reference",
        "managed_ref": ref,
        "source_repository": "kyutai/tts-voices",
        "source_revision": "pinned-revision",
        "source_path": "alba-mackenna/casual.wav",
        "source_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "license": "cc-by-4.0",
        "dataset": "alba-mackenna",
        "variant": "casual",
    }
    assert voice.identity == {
        "kind": "managed_reference",
        "ref": ref,
        "source_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_revision": "pinned-revision",
        "prepared_sha256": voice.fingerprint,
        "sample_rate": 24_000,
    }


def test_pinned_prompt_info_is_accepted_and_exposed_as_managed_identity(tmp_path: Path) -> None:
    ref = "kyutai-tts-voices:alba-mackenna/casual"
    info = VoicePromptInfo(
        ref=ref,
        source_repository="kyutai/tts-voices",
        source_revision="pinned-revision",
        source_path="alba-mackenna/casual.wav",
        size=1234,
        sha256="a" * 64,
        license="cc-by-4.0",
        dataset="alba-mackenna",
        variant="casual",
    )
    prompt = ResolvedVoicePrompt(
        ref=ref,
        path=tmp_path / "casual.wav",
        source_repository=info.source_repository,
        source_revision=info.source_revision,
        source_path=info.source_path,
        size=info.size,
        sha256=info.sha256,
        license=info.license,
        dataset=info.dataset,
        variant=info.variant,
    )
    audio = np.array([0.25, -0.5], dtype=np.float32)
    resolver = MagicMock(return_value=prompt)
    runtime = MagicMock()

    with patch("pocketsynth.voice.load_reference_audio", return_value=audio):
        voice = prepare_voice(
            runtime,
            info,
            sample_rate=24_000,
            bundle_id="english_2026-04",
            source_revision="bundle-revision",
            managed_voice_resolver=resolver,
        )

    resolver.assert_called_once_with(info)
    assert voice.voice_prompt == info
    assert voice.bundle_revision == "bundle-revision"
    assert voice.voice_prompt.source_revision == "pinned-revision"
    assert voice.metadata["kind"] == "managed_reference"
    assert voice.identity == {
        "kind": "managed_reference",
        "ref": ref,
        "source_sha256": info.sha256,
        "source_revision": info.source_revision,
        "prepared_sha256": voice.fingerprint,
        "sample_rate": 24_000,
    }


def test_managed_and_local_wav_share_fingerprint_but_keep_distinct_identity(tmp_path: Path) -> None:
    path = tmp_path / "same.wav"
    frames = np.array([-2000, 0, 1000, 2500], dtype="<i2").tobytes()
    _write_wav(path, channels=1, width=2, frames=frames, rate=16_000)
    ref = "kyutai-tts-voices:alba-mackenna/casual"
    prompt = ResolvedVoicePrompt(
        ref=ref,
        path=path,
        source_repository="kyutai/tts-voices",
        source_revision="pinned-revision",
        source_path="alba-mackenna/casual.wav",
        size=len(frames),
        sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        license="cc-by-4.0",
        dataset="alba-mackenna",
        variant="casual",
    )
    runtime = MagicMock()

    local = prepare_voice(runtime, path, sample_rate=24_000)
    managed = prepare_voice(
        runtime,
        ref,
        sample_rate=24_000,
        managed_voice_resolver=MagicMock(return_value=prompt),
    )

    assert local.fingerprint == managed.fingerprint
    assert local.identity is not None and local.identity["kind"] == "reference"
    assert managed.identity is not None and managed.identity["kind"] == "managed_reference"
    assert managed.voice_prompt is not None and managed.voice_prompt.ref == ref

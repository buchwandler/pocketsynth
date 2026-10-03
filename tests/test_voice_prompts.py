from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pocketsynth.errors import InvalidVoicePromptMetadataError
from pocketsynth.voice_prompts import (
    VoicePromptInfo,
    inspect_voice_prompt,
    list_voice_prompts,
)

SHA256 = "a" * 64


def prompt_info(**changes: object) -> VoicePromptInfo:
    values: dict[str, object] = {
        "ref": "kyutai-tts-voices:alba-mackenna/casual",
        "source_repository": "kyutai/tts-voices",
        "source_revision": "pinned-revision",
        "source_path": "alba-mackenna/casual.wav",
        "size": 1234,
        "sha256": SHA256,
        "license": "cc-by-4.0",
        "dataset": "alba-mackenna",
        "variant": "casual",
    }
    values.update(changes)
    return VoicePromptInfo(**values)  # type: ignore[arg-type]


def catalog_record(**changes: object) -> MagicMock:
    values = {
        "ref": "kyutai-tts-voices:alba-mackenna/casual",
        "source_repository": "kyutai/tts-voices",
        "source_revision": "pinned-revision",
        "source_path": "alba-mackenna/casual.wav",
        "size": 1234,
        "sha256": SHA256,
        "license": "cc-by-4.0",
        "dataset": "alba-mackenna",
        "variant": "casual",
    }
    values.update(changes)
    return MagicMock(**values)


def test_voice_prompt_info_is_validated_frozen_and_value_comparable() -> None:
    prompt = prompt_info()

    assert prompt == prompt_info()
    assert prompt.sha256 == SHA256
    with pytest.raises(FrozenInstanceError):
        prompt.ref = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ref", ""),
        ("ref", "kyutai-tts-voices:"),
        ("ref", "kyutai-tts-voices: prompt"),
        ("source_repository", " "),
        ("source_revision", ""),
        ("source_path", ""),
        ("license", ""),
        ("dataset", ""),
        ("variant", ""),
    ],
)
def test_voice_prompt_info_rejects_blank_or_noncanonical_text(field: str, value: str) -> None:
    with pytest.raises(InvalidVoicePromptMetadataError):
        prompt_info(**{field: value})


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_voice_prompt_info_rejects_invalid_size(size: object) -> None:
    with pytest.raises(InvalidVoicePromptMetadataError, match="size"):
        prompt_info(size=size)


@pytest.mark.parametrize("sha256", ["a" * 63, "g" * 64, "A" * 64])
def test_voice_prompt_info_rejects_invalid_sha256(sha256: str) -> None:
    with pytest.raises(InvalidVoicePromptMetadataError, match="sha256"):
        prompt_info(sha256=sha256)


def test_list_voice_prompts_returns_only_typed_metadata_and_forwards_options(
    tmp_path: Path,
) -> None:
    record = catalog_record()
    progress = MagicMock()
    catalog_path = tmp_path / "voice-prompts.json"

    with patch(
        "pocketsynth.voice_prompts.list_voice_prompt_metadata", return_value=(record,)
    ) as query:
        result = list_voice_prompts(
            cache_dir=tmp_path / "cache",
            catalog_path=catalog_path,
            offline=True,
            refresh=True,
            dataset="alba-mackenna",
            variant="casual",
            license="cc-by-4.0",
            progress=progress,
        )

    assert isinstance(result, tuple)
    assert result == (prompt_info(),)
    query.assert_called_once_with(
        cache_dir=tmp_path / "cache",
        catalog_path=catalog_path,
        offline=True,
        refresh=True,
        dataset="alba-mackenna",
        variant="casual",
        license="cc-by-4.0",
        progress=progress,
    )


def test_inspect_voice_prompt_returns_typed_metadata_and_forwards_options(
    tmp_path: Path,
) -> None:
    record = catalog_record()
    catalog_path = tmp_path / "voice-prompts.json"

    with patch(
        "pocketsynth.voice_prompts.inspect_voice_prompt_metadata", return_value=record
    ) as query:
        result = inspect_voice_prompt(
            "kyutai-tts-voices:alba-mackenna/casual",
            cache_dir=tmp_path / "cache",
            catalog_path=catalog_path,
            offline=True,
            refresh=True,
        )

    assert result == prompt_info()
    query.assert_called_once_with(
        "kyutai-tts-voices:alba-mackenna/casual",
        cache_dir=tmp_path / "cache",
        catalog_path=catalog_path,
        offline=True,
        refresh=True,
    )


def test_public_api_rejects_incomplete_foreign_metadata() -> None:
    record = catalog_record()
    del record.source_path

    with patch("pocketsynth.voice_prompts.inspect_voice_prompt_metadata", return_value=record):
        with pytest.raises(InvalidVoicePromptMetadataError, match="incomplete"):
            inspect_voice_prompt("kyutai-tts-voices:alba-mackenna/casual")


def test_prompt_info_replace_preserves_validation() -> None:
    prompt = prompt_info()

    with pytest.raises(InvalidVoicePromptMetadataError):
        replace(prompt, sha256="invalid")


def test_metadata_adapters_forward_options_without_fetching_prompt_audio(
    tmp_path: Path,
) -> None:
    from pocketsynth._onnxvoice import inspect_voice_prompt_metadata, list_voice_prompt_metadata

    record = catalog_record()
    module = MagicMock()
    manager = module.OnnxVoice.return_value
    manager.list_pocket_voice_prompts.return_value = (record,)
    manager.resolve_pocket_voice_prompt.return_value = record
    progress_events = []
    catalog_path = tmp_path / "voice-prompts.json"

    with patch("pocketsynth._onnxvoice._onnxvoice", return_value=module):
        listed = list_voice_prompt_metadata(
            cache_dir=tmp_path / "cache",
            catalog_path=catalog_path,
            offline=True,
            refresh=True,
            dataset="alba-mackenna",
            variant="casual",
            license="cc-by-4.0",
            progress=progress_events.append,
        )
        upstream_progress = manager.list_pocket_voice_prompts.call_args.kwargs["progress"]
        upstream_progress(type("Event", (), {"phase": "catalog_refresh", "ref": record.ref})())
        inspected = inspect_voice_prompt_metadata(
            record.ref,
            cache_dir=tmp_path / "cache",
            catalog_path=catalog_path,
            offline=True,
            refresh=True,
        )

    assert listed == (record,)
    assert inspected is record
    module.OnnxVoice.assert_called_with(
        cache_dir=tmp_path / "cache",
        catalog_sources={"pocket_voice_prompts": str(catalog_path)},
        offline=True,
    )
    manager.list_pocket_voice_prompts.assert_called_once_with(
        dataset="alba-mackenna",
        variant="casual",
        license="cc-by-4.0",
        refresh=True,
        progress=upstream_progress,
    )
    manager.resolve_pocket_voice_prompt.assert_called_once_with(record.ref, refresh=True)
    assert progress_events[0].phase == "catalog"
    assert progress_events[0].ref == record.ref
    manager.fetch_pocket_voice_prompt.assert_not_called()
    manager.install.assert_not_called()
    manager.open.assert_not_called()


@pytest.mark.parametrize(
    ("error_name", "expected_error"),
    [
        ("VoicePromptOfflineError", "OfflineAssetError"),
        ("VoicePromptIntegrityError", "AssetCacheError"),
        ("VoicePromptCatalogError", "CatalogUnavailableError"),
        ("InvalidVoicePromptRefError", "VoicePromptError"),
    ],
)
def test_metadata_inspection_maps_onnxvoice_errors(error_name: str, expected_error: str) -> None:
    from pocketsynth import errors

    error_type = type(error_name, (Exception,), {})
    module = MagicMock()
    module.OnnxVoice.return_value.resolve_pocket_voice_prompt.side_effect = error_type(
        "catalog detail"
    )

    with patch("pocketsynth._onnxvoice._onnxvoice", return_value=module):
        with pytest.raises(getattr(errors, expected_error), match="catalog detail"):
            inspect_voice_prompt("kyutai-tts-voices:missing")


def test_fetch_voice_prompt_preserves_pinned_info_and_fetches_only_after_match(
    tmp_path: Path,
) -> None:
    from pocketsynth._onnxvoice import fetch_voice_prompt

    info = prompt_info()
    record = catalog_record()
    path = tmp_path / "prompt.wav"
    progress = MagicMock()
    module = MagicMock()
    manager = module.OnnxVoice.return_value
    manager.resolve_pocket_voice_prompt.return_value = record
    manager.fetch_pocket_voice_prompt.return_value = path

    with patch("pocketsynth._onnxvoice._onnxvoice", return_value=module):
        resolved = fetch_voice_prompt(
            info, cache_dir=tmp_path / "cache", offline=True, refresh=True, progress=progress
        )

    assert resolved.path == path
    assert resolved.ref == info.ref
    assert resolved.sha256 == info.sha256
    assert resolved.source_revision == info.source_revision
    manager.resolve_pocket_voice_prompt.assert_called_once_with(info.ref, refresh=True)
    fetch_call = manager.fetch_pocket_voice_prompt.call_args
    assert fetch_call.args == (info.ref,)
    assert callable(fetch_call.kwargs["progress"])
    assert "refresh" not in fetch_call.kwargs
    module.OnnxVoice.assert_called_once_with(cache_dir=tmp_path / "cache", offline=True)


@pytest.mark.parametrize(
    ("changed", "actual_sha256", "actual_revision"),
    [
        ({"sha256": "b" * 64}, "b" * 64, "pinned-revision"),
        ({"source_revision": "new-revision"}, SHA256, "new-revision"),
    ],
)
def test_fetch_pinned_prompt_rejects_catalog_changes_before_audio_fetch(
    changed: dict[str, str], actual_sha256: str, actual_revision: str
) -> None:
    from pocketsynth._onnxvoice import fetch_voice_prompt
    from pocketsynth.errors import VoicePromptChangedError

    info = prompt_info()
    module = MagicMock()
    manager = module.OnnxVoice.return_value
    manager.resolve_pocket_voice_prompt.return_value = catalog_record(**changed)

    with patch("pocketsynth._onnxvoice._onnxvoice", return_value=module):
        with pytest.raises(VoicePromptChangedError) as caught:
            fetch_voice_prompt(info)

    assert caught.value.ref == info.ref
    assert caught.value.expected_sha256 == info.sha256
    assert caught.value.actual_sha256 == actual_sha256
    assert caught.value.expected_revision == info.source_revision
    assert caught.value.actual_revision == actual_revision
    manager.fetch_pocket_voice_prompt.assert_not_called()

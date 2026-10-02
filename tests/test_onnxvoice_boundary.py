import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pocketsynth._onnxvoice import (
    ResolvedPocketBundle,
    ResolvedVoicePrompt,
    _call,
    normalize_pocket_ref,
    normalize_provider_request,
    open_installed_bundle,
    resolve_voice_prompt,
)
from pocketsynth.errors import (
    AssetAccessError,
    AssetCacheError,
    AssetDownloadError,
    BundleNotFoundError,
    ModelInferenceError,
    OfflineAssetError,
    OptionalDependencyError,
    UnsupportedBundleError,
    VoicePromptError,
)
from pocketsynth.runtime import PocketRuntime


def test_normalize_pocket_ref():
    assert normalize_pocket_ref("english_2026-04") == "pocket:english_2026-04"
    assert normalize_pocket_ref("pocket:german") == "pocket:german"
    with pytest.raises(BundleNotFoundError):
        normalize_pocket_ref("piper:en_US-lessac-medium")


def test_provider_default_is_cpu():
    providers, options = normalize_provider_request(None, None)
    assert providers == "CPUExecutionProvider"
    assert options is None


def test_open_installed_bundle_passes_cache_and_offline_to_onnxvoice(tmp_path: Path) -> None:
    module = MagicMock()
    manager = module.OnnxVoice.return_value
    resolved = MagicMock()
    cache_dir = tmp_path / "onnxvoice-cache"

    with patch("pocketsynth._onnxvoice._onnxvoice", return_value=module):
        open_installed_bundle(
            resolved,
            providers="CPUExecutionProvider",
            cache_dir=cache_dir,
            offline=True,
        )

    module.OnnxVoice.assert_called_once_with(cache_dir=cache_dir, offline=True)
    manager.open.assert_called_once_with(
        resolved.installation,
        providers="CPUExecutionProvider",
        provider_options=None,
        session_options=None,
    )


def test_runtime_error_mapping_preserves_cause():
    class InferenceFailure(Exception):
        pass

    def fail():
        raise InferenceFailure("missing graph output 'audio'")

    with pytest.raises(ModelInferenceError, match="missing graph output 'audio'") as caught:
        _call("infer", fail)
    assert isinstance(caught.value.__cause__, InferenceFailure)


@pytest.mark.parametrize(
    ("error_name", "expected_type"),
    [
        ("PredefinedVoiceAccessError", AssetAccessError),
        ("PredefinedVoiceNotFoundError", AssetDownloadError),
        ("PredefinedVoiceIntegrityError", AssetCacheError),
        ("OfflineError", OfflineAssetError),
    ],
)
def test_predefined_voice_errors_remain_actionable(error_name, expected_type) -> None:
    error_type = type(error_name, (Exception,), {})

    def fail() -> None:
        raise error_type("safe error detail")

    with pytest.raises(expected_type, match="safe error detail") as caught:
        _call("prepare_voice", fail)
    assert isinstance(caught.value.__cause__, error_type)


def test_huggingface_dependency_error_explains_pocketsynth_install() -> None:
    error_type = type("OptionalDependencyError", (Exception,), {})

    def fail() -> None:
        raise error_type(
            "Remote Pocket downloads require Hugging Face support. Install 'onnxvoice[pocket]'."
        )

    with pytest.raises(
        OptionalDependencyError,
        match=r"pocketsynth\[cpu\].*pocketsynth\[gpu\]",
    ):
        _call("install", fail)


def test_managed_bundle_catalog_voice_names_reach_runtime(tmp_path: Path) -> None:
    bundle = {
        "bundle_name": "english_2026-04",
        "language": "en",
        "schema_version": 2,
        "sample_rate": 24000,
        "samples_per_frame": 1920,
        "max_token_per_chunk": 50,
        "tokenizer_file": "tokenizer.model",
        "bos_before_voice_file": "bos.npy",
    }
    metadata_path = tmp_path / "bundle.json"
    metadata_path.write_text(json.dumps(bundle), encoding="utf-8")
    resolved = ResolvedPocketBundle(
        ref="pocket:english_2026-04",
        bundle_id="english_2026-04",
        path=tmp_path,
        tokenizer_path=tmp_path / "tokenizer.model",
        metadata_path=metadata_path,
        precision="int8",
        source_revision="catalog-revision",
        metadata={"predefined_voice_names": ["alba"]},
    )

    cache_dir = tmp_path / "onnxvoice-cache"
    progress = MagicMock()
    adapter = MagicMock()
    state = object()
    adapter.prepare_predefined_voice.return_value = state
    with (
        patch("pocketsynth.runtime.open_installed_bundle", return_value=adapter) as open_runtime,
        patch("pocketsynth.runtime.PocketFrontend"),
    ):
        runtime = PocketRuntime.from_resolved(
            resolved, cache_dir=cache_dir, offline=True, progress=progress
        )

    assert runtime._cache_dir == cache_dir
    assert runtime._offline is True
    assert runtime._progress is progress
    with patch("pocketsynth.runtime.resolve_voice_prompt") as resolve_prompt:
        runtime._resolve_voice_prompt("kyutai-tts-voices:alba-mackenna/casual")
    resolve_prompt.assert_called_once_with(
        "kyutai-tts-voices:alba-mackenna/casual",
        cache_dir=cache_dir,
        offline=True,
        progress=progress,
    )
    with patch(
        "pocketsynth.runtime.list_pocket_voice_prompts", return_value=("prompt",)
    ) as list_prompts:
        assert runtime.list_voice_prompts(
            dataset="alba-mackenna", variant="casual", license="cc-by-4.0"
        ) == ("prompt",)
    list_prompts.assert_called_once_with(
        cache_dir=cache_dir,
        offline=True,
        dataset="alba-mackenna",
        variant="casual",
        license="cc-by-4.0",
        progress=progress,
    )
    assert runtime.predefined_voices == ("alba",)
    open_runtime.assert_called_once_with(
        resolved,
        providers=None,
        provider_options=None,
        session_options=None,
        cache_dir=cache_dir,
        offline=True,
    )
    voice = runtime.prepare_voice("alba")
    adapter.prepare_predefined_voice.assert_called_once_with("alba")
    adapter.prepare_voice.assert_not_called()
    assert voice.state is state
    assert voice.metadata == {
        "kind": "predefined",
        "name": "alba",
        "source_revision": "catalog-revision",
    }
    assert voice.bundle_id == "english_2026-04"


def test_managed_bundle_rejects_catalog_voice_name_disagreement(tmp_path: Path) -> None:
    bundle = {
        "bundle_name": "english_2026-04",
        "language": "en",
        "schema_version": 2,
        "sample_rate": 24000,
        "samples_per_frame": 1920,
        "max_token_per_chunk": 50,
        "tokenizer_file": "tokenizer.model",
        "bos_before_voice_file": "bos.npy",
        "predefined_voices": ["alba"],
    }
    metadata_path = tmp_path / "bundle.json"
    metadata_path.write_text(json.dumps(bundle), encoding="utf-8")
    resolved = ResolvedPocketBundle(
        ref="pocket:english_2026-04",
        bundle_id="english_2026-04",
        path=tmp_path,
        tokenizer_path=tmp_path / "tokenizer.model",
        metadata_path=metadata_path,
        precision="int8",
        source_revision="catalog-revision",
        metadata={"predefined_voice_names": ["other"]},
    )

    with pytest.raises(UnsupportedBundleError, match="disagree"):
        PocketRuntime.from_resolved(resolved)


def test_resolve_voice_prompt_resolves_and_fetches_through_onnxvoice(
    tmp_path: Path,
) -> None:
    ref = "kyutai-tts-voices:alba-mackenna/casual"
    path = tmp_path / "casual.wav"
    prompt = MagicMock(
        ref=ref,
        source_repository="kyutai/tts-voices",
        source_revision="pinned-revision",
        source_path="alba-mackenna/casual.wav",
        size=1234,
        sha256="prompt-sha256",
        license="cc-by-4.0",
        dataset="alba-mackenna",
        variant="casual",
    )
    module = MagicMock()
    manager = module.OnnxVoice.return_value
    manager.resolve_pocket_voice_prompt.return_value = prompt
    manager.fetch_pocket_voice_prompt.return_value = path
    progress_events = []
    progress = progress_events.append
    cache_dir = tmp_path / "cache"

    with patch("pocketsynth._onnxvoice._onnxvoice", return_value=module):
        resolved = resolve_voice_prompt(ref, cache_dir=cache_dir, offline=True, progress=progress)

    module.OnnxVoice.assert_called_once_with(cache_dir=cache_dir, offline=True)
    manager.resolve_pocket_voice_prompt.assert_called_once_with(ref)
    manager.fetch_pocket_voice_prompt.assert_called_once()
    assert manager.fetch_pocket_voice_prompt.call_args.args == (ref,)
    upstream_progress = manager.fetch_pocket_voice_prompt.call_args.kwargs["progress"]
    assert callable(upstream_progress)
    upstream_progress(type("Event", (), {"phase": "download_started", "ref": ref})())
    assert progress_events[0].phase == "download"
    assert progress_events[0].ref == ref
    assert resolved == ResolvedVoicePrompt(
        ref=ref,
        path=path,
        source_repository="kyutai/tts-voices",
        source_revision="pinned-revision",
        source_path="alba-mackenna/casual.wav",
        size=1234,
        sha256="prompt-sha256",
        license="cc-by-4.0",
        dataset="alba-mackenna",
        variant="casual",
    )


@pytest.mark.parametrize(
    ("upstream_name", "expected_type"),
    [
        ("VoicePromptOfflineError", OfflineAssetError),
        ("VoicePromptIntegrityError", AssetCacheError),
        ("UnknownVoicePromptError", VoicePromptError),
    ],
)
def test_managed_prompt_errors_map_to_actionable_pocketsynth_errors(
    upstream_name: str, expected_type: type[Exception]
) -> None:
    error_type = type(upstream_name, (Exception,), {})
    module = MagicMock()
    module.OnnxVoice.return_value.resolve_pocket_voice_prompt.side_effect = error_type(
        "catalog detail"
    )

    with patch("pocketsynth._onnxvoice._onnxvoice", return_value=module):
        with pytest.raises(expected_type, match="catalog detail") as caught:
            resolve_voice_prompt("kyutai-tts-voices:missing")

    assert isinstance(caught.value.__cause__, error_type)

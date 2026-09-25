import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pocketsynth._onnxvoice import (
    ResolvedPocketBundle,
    _call,
    normalize_pocket_ref,
    normalize_provider_request,
    open_installed_bundle,
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
    adapter = MagicMock()
    state = object()
    adapter.prepare_predefined_voice.return_value = state
    with (
        patch("pocketsynth.runtime.open_installed_bundle", return_value=adapter) as open_runtime,
        patch("pocketsynth.runtime.PocketFrontend"),
    ):
        runtime = PocketRuntime.from_resolved(resolved, cache_dir=cache_dir, offline=True)

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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pocketsynth import __version__
from pocketsynth.assets import PocketBundle
from pocketsynth.discovery import (
    DiscoveredBundle,
    DiscoveredVoice,
    discover_bundles,
    runtime_identity,
)


def catalog_item() -> SimpleNamespace:
    return SimpleNamespace(
        id="english_2026-04",
        ref="pocket:english_2026-04",
        aliases=("english", "en"),
        sample_rate=24_000,
        voices=("alba",),
        default_voice="alba",
        artifacts=(SimpleNamespace(quality="fp16"),),
        metadata={
            "display_name": "English 2026-04",
            "language": "en",
            "profiles": {"int8": {}, "fp32": {}},
            "predefined_voice_names": ["alba"],
            "source_revision": "bundle-revision",
            "max_token_per_chunk": 512,
            "voice_details": [
                {
                    "id": "alba",
                    "gender": "female",
                    "language": "en",
                    "locale": "en-US",
                    "language_label": "English",
                },
                {"id": "unknown-voice"},
            ],
        },
    )


def test_discover_bundles_returns_normalized_pocketsynth_owned_metadata(tmp_path: Path) -> None:
    item = catalog_item()
    progress = MagicMock()
    catalog_path = tmp_path / "catalog.json"

    with patch("pocketsynth.discovery.list_pocket_bundles", return_value=(item,)) as query:
        result = discover_bundles(
            language="en",
            cache_dir=tmp_path / "cache",
            catalog_path=catalog_path,
            offline=True,
            refresh=True,
            progress=progress,
        )

    assert isinstance(result, tuple)
    assert isinstance(result[0], DiscoveredBundle)
    assert result[0] == DiscoveredBundle(
        id="english_2026-04",
        ref="pocket:english_2026-04",
        display_name="English 2026-04",
        aliases=("english", "en"),
        language="en",
        sample_rate=24_000,
        precisions=("fp16", "fp32", "int8"),
        predefined_voices=("alba",),
        default_voice="alba",
        source_revision="bundle-revision",
        max_tokens=512,
        voice_details=(
            DiscoveredVoice(
                id="alba",
                gender="female",
                language="en",
                locale="en-US",
                language_label="English",
            ),
            DiscoveredVoice(id="unknown-voice"),
        ),
        metadata=item.metadata,
    )
    query.assert_called_once_with(
        language="en",
        cache_dir=tmp_path / "cache",
        catalog_path=catalog_path,
        offline=True,
        refresh=True,
        progress=progress,
    )


def test_bundle_discovery_adapter_only_lists_catalog_metadata(tmp_path: Path) -> None:
    from pocketsynth._onnxvoice import list_pocket_bundles

    item = catalog_item()
    module = MagicMock()
    manager = module.OnnxVoice.return_value
    manager.list.return_value = [item]
    events = []
    catalog_path = tmp_path / "catalog.json"

    with patch("pocketsynth._onnxvoice._onnxvoice", return_value=module):
        result = list_pocket_bundles(
            language="de",
            cache_dir=tmp_path / "cache",
            catalog_path=catalog_path,
            offline=True,
            refresh=True,
            progress=events.append,
        )

    assert result == (item,)
    module.OnnxVoice.assert_called_once_with(
        cache_dir=tmp_path / "cache",
        offline=True,
        catalog_sources={"pocket": str(catalog_path)},
    )
    manager.list.assert_called_once()
    args = manager.list.call_args
    assert args.args == ("pocket",)
    assert args.kwargs["language"] == "de"
    assert args.kwargs["refresh"] is True
    assert callable(args.kwargs["progress"])
    args.kwargs["progress"](SimpleNamespace(phase="catalog_refresh", ref="pocket:english"))
    assert events[0].phase == "catalog"
    assert events[0].ref == "pocket:english"
    manager.install.assert_not_called()
    manager.open.assert_not_called()


def test_runtime_identity_uses_versions_and_separate_bundle_revision() -> None:
    bundle = DiscoveredBundle(
        id="english_2026-04",
        ref="pocket:english_2026-04",
        display_name="English",
        aliases=(),
        language="en",
        sample_rate=24_000,
        precisions=("int8",),
        predefined_voices=("alba",),
        default_voice="alba",
        source_revision="bundle-revision",
        max_tokens=512,
    )

    with patch("pocketsynth.discovery.version", return_value="0.2.2"):
        identity = runtime_identity(bundle)

    assert identity == {
        "engine_version": __version__,
        "runtime_revision": "0.2.2",
        "catalog_revision": None,
        "bundle_revision": "bundle-revision",
    }


def test_runtime_identity_supports_installed_bundle_and_no_bundle() -> None:
    bundle = PocketBundle(
        bundle_id="english_2026-04",
        path=Path("bundle"),
        tokenizer_path=Path("tokenizer.model"),
        metadata_path=Path("metadata.json"),
        source_revision="installed-revision",
    )

    with patch("pocketsynth.discovery.version", return_value="0.2.2"):
        with_bundle = runtime_identity(bundle)
        without_bundle = runtime_identity()

    assert with_bundle["bundle_revision"] == "installed-revision"
    assert without_bundle["bundle_revision"] is None

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ._onnxvoice import list_pocket_bundles
from .asset_progress import AssetProgressCallback
from .assets import PocketBundle


@dataclass(frozen=True, slots=True)
class DiscoveredVoice:
    """Catalog metadata for a predefined voice declared by a Pocket bundle."""

    id: str
    gender: str = "unknown"
    language: str = "unknown"
    locale: str = "unknown"
    language_label: str = "unknown"


@dataclass(frozen=True, slots=True)
class DiscoveredBundle:
    """PocketSynth-owned metadata for a cataloged Pocket bundle."""

    id: str
    ref: str
    display_name: str
    aliases: tuple[str, ...]
    language: str
    sample_rate: int | None
    precisions: tuple[str, ...]
    predefined_voices: tuple[str, ...]
    default_voice: str | None
    source_revision: str | None
    max_tokens: int | None
    voice_details: tuple[DiscoveredVoice, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _voice_details(metadata: Mapping[str, Any]) -> tuple[DiscoveredVoice, ...]:
    raw_details = metadata.get("voice_details", ())
    if not isinstance(raw_details, Sequence) or isinstance(raw_details, (str, bytes)):
        return ()
    return tuple(
        DiscoveredVoice(
            id=detail["id"],
            gender=detail.get("gender") or "unknown",
            language=detail.get("language") or "unknown",
            locale=detail.get("locale") or "unknown",
            language_label=detail.get("language_label") or "unknown",
        )
        for detail in raw_details
        if isinstance(detail, Mapping) and isinstance(detail.get("id"), str)
    )


def _from_catalog_item(item: Any) -> DiscoveredBundle:
    metadata = dict(item.metadata or {})
    profiles = metadata.get("profiles")
    precisions = (
        {quality for quality in profiles if isinstance(quality, str)}
        if isinstance(profiles, Mapping)
        else set()
    )
    precisions.update(
        quality
        for artifact in item.artifacts
        if isinstance((quality := artifact.quality), str) and quality
    )
    raw_voices = metadata.get("predefined_voice_names")
    if isinstance(raw_voices, Sequence) and not isinstance(raw_voices, (str, bytes)):
        predefined_voices = tuple(name for name in raw_voices if isinstance(name, str))
    else:
        predefined_voices = tuple(item.voices)
    display_name = metadata.get("display_name") or metadata.get("name") or item.id
    language = metadata.get("language")
    max_tokens = metadata.get("max_token_per_chunk")
    return DiscoveredBundle(
        id=item.id,
        ref=item.ref,
        display_name=display_name if isinstance(display_name, str) else item.id,
        aliases=tuple(item.aliases),
        language=language if isinstance(language, str) and language else "unknown",
        sample_rate=item.sample_rate,
        precisions=tuple(sorted(precisions)),
        predefined_voices=predefined_voices,
        default_voice=(
            item.default_voice
            if isinstance(item.default_voice, str) and item.default_voice
            else None
        ),
        source_revision=(
            metadata.get("source_revision")
            if isinstance(metadata.get("source_revision"), str)
            else None
        ),
        max_tokens=(
            max_tokens if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) else None
        ),
        voice_details=_voice_details(metadata),
        metadata=metadata,
    )


def discover_bundles(
    *,
    language: str | None = None,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    progress: AssetProgressCallback | None = None,
) -> tuple[DiscoveredBundle, ...]:
    """List Pocket bundle metadata without installing bundles or opening a runtime."""
    items = list_pocket_bundles(
        language=language,
        cache_dir=cache_dir,
        catalog_path=catalog_path,
        offline=offline,
        refresh=refresh,
        progress=progress,
    )
    return tuple(_from_catalog_item(item) for item in items)


def runtime_identity(
    bundle: DiscoveredBundle | PocketBundle | None = None,
) -> dict[str, str | None]:
    """Return PocketSynth and OnnxVoice version identity without opening a model."""
    from . import __version__

    return {
        "engine_version": __version__,
        "runtime_revision": version("onnxvoice"),
        "catalog_revision": None,
        "bundle_revision": bundle.source_revision if bundle is not None else None,
    }

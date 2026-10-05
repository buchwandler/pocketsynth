from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._onnxvoice import list_pocket_bundles
from .asset_progress import AssetProgressCallback
from .assets import PocketBundle


def _require_nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of non-empty strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{name} must contain only non-empty strings")
    return result


@dataclass(frozen=True, slots=True)
class DiscoveredVoice:
    """Catalog metadata for a predefined voice declared by a Pocket bundle."""

    id: str
    gender: str = "unknown"
    language: str = "unknown"
    locale: str = "unknown"
    language_label: str = "unknown"

    def __post_init__(self) -> None:
        for name in ("id", "gender", "language", "locale", "language_label"):
            _require_nonempty_string(getattr(self, name), name)


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

    def __post_init__(self) -> None:
        for name in ("id", "ref", "display_name", "language"):
            _require_nonempty_string(getattr(self, name), name)

        for name in ("aliases", "precisions", "predefined_voices"):
            object.__setattr__(self, name, _string_tuple(getattr(self, name), name))

        if self.sample_rate is not None and (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a positive integer or None")
        if self.default_voice is not None:
            _require_nonempty_string(self.default_voice, "default_voice")
        if self.source_revision is not None:
            _require_nonempty_string(self.source_revision, "source_revision")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer or None")

        if not isinstance(self.voice_details, Sequence) or isinstance(
            self.voice_details, (str, bytes)
        ):
            raise ValueError("voice_details must be a sequence of DiscoveredVoice values")
        voice_details = tuple(self.voice_details)
        if any(not isinstance(item, DiscoveredVoice) for item in voice_details):
            raise ValueError("voice_details must contain only DiscoveredVoice values")
        object.__setattr__(self, "voice_details", voice_details)

        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

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
    from .api_contract import REQUEST_API_VERSION

    return {
        "engine_version": __version__,
        "runtime_revision": version("onnxvoice"),
        "request_api_version": str(REQUEST_API_VERSION),
        "catalog_revision": None,
        "bundle_revision": bundle.source_revision if bundle is not None else None,
    }

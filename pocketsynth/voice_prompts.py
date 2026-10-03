from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._onnxvoice import inspect_voice_prompt_metadata, list_voice_prompt_metadata
from .asset_progress import AssetProgressCallback
from .errors import InvalidVoicePromptMetadataError

_PROMPT_REF_PREFIX = "kyutai-tts-voices:"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class VoicePromptInfo:
    """Pinned catalog metadata for one managed voice prompt."""

    ref: str
    source_repository: str
    source_revision: str
    source_path: str
    size: int
    sha256: str
    license: str
    dataset: str
    variant: str

    def __post_init__(self) -> None:
        for name in (
            "ref",
            "source_repository",
            "source_revision",
            "source_path",
            "license",
            "dataset",
            "variant",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidVoicePromptMetadataError(f"{name} must be a non-empty string")
        prompt_id = self.ref.removeprefix(_PROMPT_REF_PREFIX)
        if (
            not self.ref.startswith(_PROMPT_REF_PREFIX)
            or self.ref != self.ref.strip()
            or not prompt_id
            or prompt_id != prompt_id.strip()
        ):
            raise InvalidVoicePromptMetadataError(
                f"ref must be a canonical {_PROMPT_REF_PREFIX!r} reference"
            )
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise InvalidVoicePromptMetadataError("size must be a positive integer")
        if not isinstance(self.sha256, str) or _SHA256_RE.fullmatch(self.sha256) is None:
            raise InvalidVoicePromptMetadataError(
                "sha256 must be a lowercase 64-character hex digest"
            )


def _from_catalog_record(record: Any) -> VoicePromptInfo:
    try:
        return VoicePromptInfo(
            ref=record.ref,
            source_repository=record.source_repository,
            source_revision=record.source_revision,
            source_path=record.source_path,
            size=record.size,
            sha256=record.sha256,
            license=record.license,
            dataset=record.dataset,
            variant=record.variant,
        )
    except AttributeError as exc:
        raise InvalidVoicePromptMetadataError(
            "OnnxVoice returned incomplete voice prompt metadata"
        ) from exc


def list_voice_prompts(
    *,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    dataset: str | None = None,
    variant: str | None = None,
    license: str | None = None,
    progress: AssetProgressCallback | None = None,
) -> tuple[VoicePromptInfo, ...]:
    """List managed prompt metadata without fetching prompt audio or opening a runtime."""
    records = list_voice_prompt_metadata(
        cache_dir=cache_dir,
        catalog_path=catalog_path,
        offline=offline,
        refresh=refresh,
        dataset=dataset,
        variant=variant,
        license=license,
        progress=progress,
    )
    return tuple(_from_catalog_record(record) for record in records)


def inspect_voice_prompt(
    ref: str,
    *,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> VoicePromptInfo:
    """Resolve one managed prompt's pinned metadata without fetching its audio."""
    record = inspect_voice_prompt_metadata(
        ref,
        cache_dir=cache_dir,
        catalog_path=catalog_path,
        offline=offline,
        refresh=refresh,
    )
    return _from_catalog_record(record)

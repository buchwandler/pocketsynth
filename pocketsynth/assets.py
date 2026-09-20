from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._onnxvoice import ResolvedPocketBundle, installation_to_bundle_info


@dataclass(frozen=True, slots=True)
class PocketBundle:
    bundle_id: str | None
    path: Path
    tokenizer_path: Path
    metadata_path: Path
    precision: str | None = None
    source_revision: str | None = None
    metadata: Mapping[str, Any] | None = None
    installation: Any | None = None

    @classmethod
    def from_resolved(cls, resolved: ResolvedPocketBundle) -> PocketBundle:
        return cls(
            resolved.bundle_id,
            resolved.path,
            resolved.tokenizer_path,
            resolved.metadata_path,
            resolved.precision,
            resolved.source_revision,
            resolved.metadata,
            resolved.installation,
        )

    @classmethod
    def from_installation(cls, installation: Any) -> PocketBundle:
        return cls.from_resolved(installation_to_bundle_info(installation))

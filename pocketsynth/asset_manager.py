from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ._onnxvoice import normalize_pocket_ref
from .assets import PocketBundle
from .bundle import Precision
from .errors import OptionalDependencyError


def _offline_value(value: bool | None) -> bool:
    if value is not None:
        return value
    return os.environ.get("POCKETSYNTH_OFFLINE", "").casefold() in {"1", "true", "yes", "on"}


class BundleAssetManager:
    """Thin compatibility facade over OnnxVoice's Pocket catalog/store."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        catalog_path: str | Path | None = None,
        offline: bool | None = None,
        progress: Any | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.catalog_path = Path(catalog_path) if catalog_path is not None else None
        self.offline = _offline_value(offline)
        self.progress = progress

    def _manager(self) -> Any:
        try:
            import onnxvoice
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError("OnnxVoice is required for Pocket bundle assets") from exc
        sources = {"pocket": str(self.catalog_path)} if self.catalog_path is not None else None
        return onnxvoice.OnnxVoice(
            cache_dir=self.cache_dir, catalog_sources=sources, offline=self.offline
        )

    def list_bundles(
        self, *, language: str | None = None, refresh: bool = False
    ) -> tuple[Any, ...]:
        return tuple(
            self._manager().list(
                "pocket", language=language, refresh=refresh, progress=self.progress
            )
        )

    def resolve_bundle(
        self,
        bundle: str,
        *,
        precision: Precision = "int8",
        download: bool = True,
        refresh_catalog: bool = False,
        force_download: bool = False,
    ) -> PocketBundle:
        manager = self._manager()
        ref = normalize_pocket_ref(bundle)
        installation = (
            manager.install(
                ref,
                quality=precision,
                refresh=refresh_catalog,
                force=force_download,
                progress=self.progress,
            )
            if download
            else manager.resolve(ref, quality=precision)
        )
        return PocketBundle.from_installation(installation)

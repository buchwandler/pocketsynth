from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_progress import AssetProgressCallback, adapt_asset_progress
from .bundle import BundlePaths, Precision
from .errors import (
    AssetCacheError,
    AssetDownloadError,
    AssetError,
    BundleNotFoundError,
    CatalogUnavailableError,
    ModelInferenceError,
    OfflineAssetError,
    OptionalDependencyError,
    RuntimeCapabilityError,
    SessionCreationError,
    UnsupportedBundleError,
)


@dataclass(frozen=True, slots=True)
class ResolvedPocketBundle:
    ref: str | None
    bundle_id: str | None
    path: Path
    tokenizer_path: Path
    metadata_path: Path
    precision: str | None
    source_revision: str | None
    metadata: Mapping[str, Any]
    installation: Any | None = None


def _onnxvoice() -> Any:
    try:
        import onnxvoice
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "OnnxVoice with Pocket support is required. Install pocketsynth[cpu] or pocketsynth[gpu]."
        ) from exc
    return onnxvoice


def normalize_pocket_ref(ref: str) -> str:
    if not isinstance(ref, str) or not ref.strip():
        raise BundleNotFoundError("Pocket bundle reference must not be empty")
    value = ref.strip()
    if ":" not in value:
        return f"pocket:{value}"
    system, item_id = value.split(":", 1)
    if system.casefold() != "pocket" or not item_id:
        raise BundleNotFoundError(f"Not a Pocket bundle reference: {ref!r}")
    return f"pocket:{item_id}"


def normalize_provider_request(
    providers: Sequence[Any] | str | None,
    provider_options: Mapping[str, Any] | None,
) -> tuple[str | Sequence[str] | None, Any | None]:
    if providers is None:
        return "CPUExecutionProvider", provider_options
    if isinstance(providers, str):
        return providers, provider_options
    names: list[str] = []
    options: list[dict[str, Any]] = []
    explicit = False
    for provider in providers:
        if hasattr(provider, "name"):
            names.append(str(provider.name))
            value = dict(getattr(provider, "options", {}) or {})
            options.append(value)
            explicit = explicit or bool(value)
        elif isinstance(provider, tuple):
            name, raw = provider
            names.append(str(name))
            options.append(dict(raw or {}))
            explicit = True
        else:
            names.append(str(provider))
            options.append(dict(provider_options or {}))
    if not names:
        raise ValueError("providers must not be empty")
    return names, options if explicit else provider_options


def _map_error(exc: Exception, *, operation: str) -> Exception:
    name = type(exc).__name__
    message = str(exc) or name
    if name in {"AssetNotFoundError", "NotInstalledError"}:
        return BundleNotFoundError(message)
    if name == "OfflineError":
        return OfflineAssetError(message)
    if name in {"IntegrityError", "ManifestError", "UnsafePathError", "LockError"}:
        return AssetCacheError(message)
    if name == "CatalogError":
        return CatalogUnavailableError(message)
    if name in {"RuntimeContractError", "CapabilityError", "UnsupportedSystemError"}:
        return RuntimeCapabilityError(message)
    if operation == "infer":
        return ModelInferenceError(f"Pocket ONNX inference failed: {message}")
    if operation == "open":
        return SessionCreationError(f"Could not open Pocket ONNX runtime: {message}")
    return AssetDownloadError(message) if operation == "install" else AssetError(message)


def _call(operation: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except (FileNotFoundError, ValueError, TypeError):
        raise
    except Exception as exc:
        mapped = _map_error(exc, operation=operation)
        raise mapped from exc


def open_local_bundle(
    paths: BundlePaths,
    *,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
) -> Any:
    requested, options = normalize_provider_request(providers, provider_options)
    module = _onnxvoice()
    try:
        return _call(
            "open",
            lambda: module.open_local(
                system="pocket",
                files=paths.runtime_files(),
                metadata={
                    "bundle_name": paths.metadata.bundle_name,
                    "selected_quality": paths.precision,
                    "managed": False,
                },
                sample_rate=paths.metadata.sample_rate,
                providers=requested,
                provider_options=options,
                session_options=session_options,
            ),
        )
    except TypeError as exc:
        raise RuntimeCapabilityError(
            "Installed OnnxVoice does not yet support open_local(files=...) for Pocket bundles. "
            "Apply the accompanying OnnxVoice implementation brief."
        ) from exc


def install_pretrained_bundle(
    ref: str,
    *,
    precision: Precision = "int8",
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh_catalog: bool = False,
    force_download: bool = False,
    progress: AssetProgressCallback | None = None,
) -> ResolvedPocketBundle:
    module = _onnxvoice()
    manager = module.OnnxVoice(cache_dir=cache_dir, offline=bool(offline))
    normalized = normalize_pocket_ref(ref)
    installation = _call(
        "install",
        lambda: manager.install(
            normalized,
            quality=precision,
            refresh=refresh_catalog,
            force=force_download,
            progress=adapt_asset_progress(progress),
        ),
    )
    return installation_to_bundle_info(installation, ref=normalized)


def open_installed_bundle(
    resolved: ResolvedPocketBundle,
    *,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
) -> Any:
    requested, options = normalize_provider_request(providers, provider_options)
    module = _onnxvoice()
    manager = module.OnnxVoice()
    return _call(
        "open",
        lambda: manager.open(
            resolved.installation,
            providers=requested,
            provider_options=options,
            session_options=session_options,
        ),
    )


def installation_to_bundle_info(
    installation: Any, *, ref: str | None = None
) -> ResolvedPocketBundle:
    try:
        metadata_path = Path(installation.artifact("bundle_metadata").path)
        tokenizer_path = Path(installation.artifact("tokenizer").path)
    except (KeyError, AttributeError) as exc:
        raise UnsupportedBundleError(
            "Pocket installation must contain bundle_metadata and tokenizer artifacts"
        ) from exc
    raw = dict(getattr(installation, "metadata", {}) or {})
    return ResolvedPocketBundle(
        ref=ref or getattr(installation, "ref", None),
        bundle_id=getattr(installation, "id", None),
        path=Path(installation.path),
        tokenizer_path=tokenizer_path,
        metadata_path=metadata_path,
        precision=(str(raw["selected_quality"]) if raw.get("selected_quality") else None),
        source_revision=(str(raw["source_revision"]) if raw.get("source_revision") else None),
        metadata=raw,
        installation=installation,
    )


def runtime_diagnostics(runtime: Any) -> dict[str, Any]:
    sessions = getattr(runtime, "sessions", None)
    if callable(sessions):
        sessions = sessions()
    return {
        "providers_requested": tuple(getattr(runtime, "providers", ()) or ()),
        "sessions": tuple(sorted(sessions)) if isinstance(sessions, Mapping) else (),
    }

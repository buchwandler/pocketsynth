from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_progress import AssetProgressCallback, adapt_asset_progress
from .bundle import BundlePaths, Precision
from .errors import (
    AssetAccessError,
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
    VoicePromptChangedError,
    VoicePromptError,
)

_ONNXVOICE_MINIMUM_VERSION = "0.2.2"


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


@dataclass(frozen=True, slots=True)
class ResolvedVoicePrompt:
    ref: str
    path: Path
    source_repository: str
    source_revision: str
    source_path: str
    size: int
    sha256: str
    license: str
    dataset: str
    variant: str


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
    if name in {"VoicePromptAccessError"}:
        return AssetAccessError(message)
    if name == "VoicePromptOfflineError":
        return OfflineAssetError(message)
    if name == "VoicePromptIntegrityError":
        return AssetCacheError(message)
    if name == "VoicePromptCatalogError":
        return CatalogUnavailableError(message)
    if name in {
        "VoicePromptError",
        "InvalidVoicePromptRefError",
        "UnknownVoicePromptError",
        "VoicePromptFormatError",
    }:
        return VoicePromptError(message)
    if name in {"VoicePromptDownloadError", "VoicePromptNotFoundError"}:
        return AssetDownloadError(message)
    if name in {
        "PredefinedVoiceAccessError",
        "AssetAccessError",
        "AssetAuthenticationError",
        "AssetPermissionError",
    }:
        return AssetAccessError(message)
    if name in {"AssetDownloadError", "PredefinedVoiceNotFoundError"}:
        return AssetDownloadError(message)
    if name == "PredefinedVoiceIntegrityError":
        return AssetCacheError(message)
    if name == "PredefinedVoiceError":
        return AssetError(message)
    if name == "OptionalDependencyError":
        if any(
            value in message.lower()
            for value in ("huggingface_hub", "onnxvoice[pocket]", "safetensors")
        ):
            message += " For PocketSynth, install pocketsynth[cpu] or pocketsynth[gpu]."
        return OptionalDependencyError(message)
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
            f"Upgrade OnnxVoice to >={_ONNXVOICE_MINIMUM_VERSION} for local Pocket bundle support."
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
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> Any:
    requested, options = normalize_provider_request(providers, provider_options)
    module = _onnxvoice()
    manager = module.OnnxVoice(cache_dir=cache_dir, offline=offline)
    return _call(
        "open",
        lambda: manager.open(
            resolved.installation,
            providers=requested,
            provider_options=options,
            session_options=session_options,
        ),
    )


def list_pocket_bundles(
    *,
    language: str | None = None,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    progress: AssetProgressCallback | None = None,
) -> tuple[Any, ...]:
    """Query Pocket catalog metadata without installing a bundle or opening a runtime."""
    module = _onnxvoice()
    kwargs: dict[str, Any] = {"cache_dir": cache_dir, "offline": offline}
    if catalog_path is not None:
        kwargs["catalog_sources"] = {"pocket": str(catalog_path)}
    manager = module.OnnxVoice(**kwargs)
    return tuple(
        _call(
            "bundle_discover",
            lambda: manager.list(
                "pocket",
                language=language,
                refresh=refresh,
                progress=adapt_asset_progress(progress),
            ),
        )
    )


def _voice_prompt_manager(
    *,
    cache_dir: str | Path | None,
    catalog_path: str | Path | None,
    offline: bool,
) -> Any:
    module = _onnxvoice()
    kwargs: dict[str, Any] = {"cache_dir": cache_dir, "offline": offline}
    if catalog_path is not None:
        kwargs["catalog_sources"] = {"pocket_voice_prompts": str(catalog_path)}
    return module.OnnxVoice(**kwargs)


def inspect_voice_prompt_metadata(
    ref: str,
    *,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> Any:
    """Resolve prompt catalog metadata without fetching its WAV payload."""
    manager = _voice_prompt_manager(
        cache_dir=cache_dir,
        catalog_path=catalog_path,
        offline=offline,
    )
    return _call(
        "voice_prompt_inspect",
        lambda: manager.resolve_pocket_voice_prompt(ref, refresh=refresh),
    )


def list_voice_prompt_metadata(
    *,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    dataset: str | None = None,
    variant: str | None = None,
    license: str | None = None,
    progress: AssetProgressCallback | None = None,
) -> tuple[Any, ...]:
    """List prompt catalog metadata without fetching WAV payloads."""
    manager = _voice_prompt_manager(
        cache_dir=cache_dir,
        catalog_path=catalog_path,
        offline=offline,
    )
    return _call(
        "voice_prompt_list",
        lambda: manager.list_pocket_voice_prompts(
            dataset=dataset,
            variant=variant,
            license=license,
            refresh=refresh,
            progress=adapt_asset_progress(progress),
        ),
    )


def fetch_voice_prompt(
    source: Any,
    *,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    progress: AssetProgressCallback | None = None,
) -> ResolvedVoicePrompt:
    """Resolve, verify, and fetch a managed prompt for runtime preparation."""
    from .voice_prompts import VoicePromptInfo, _from_catalog_record

    if isinstance(source, VoicePromptInfo):
        expected = source
        ref = source.ref
    elif isinstance(source, str):
        expected = None
        ref = source
    else:
        raise TypeError("source must be a prompt reference or VoicePromptInfo")

    manager = _voice_prompt_manager(
        cache_dir=cache_dir,
        catalog_path=catalog_path,
        offline=offline,
    )
    record = _call(
        "voice_prompt_inspect",
        lambda: manager.resolve_pocket_voice_prompt(ref, refresh=refresh),
    )
    info = _from_catalog_record(record)
    if expected is not None and info != expected:
        raise VoicePromptChangedError(
            ref=ref,
            expected_sha256=expected.sha256,
            actual_sha256=info.sha256,
            expected_revision=expected.source_revision,
            actual_revision=info.source_revision,
        )

    fetch_kwargs: dict[str, Any] = {"progress": adapt_asset_progress(progress)}
    path = _call(
        "voice_prompt_fetch",
        lambda: manager.fetch_pocket_voice_prompt(ref, **fetch_kwargs),
    )
    return ResolvedVoicePrompt(
        ref=info.ref,
        path=Path(path),
        source_repository=info.source_repository,
        source_revision=info.source_revision,
        source_path=info.source_path,
        size=info.size,
        sha256=info.sha256,
        license=info.license,
        dataset=info.dataset,
        variant=info.variant,
    )


def resolve_voice_prompt(
    ref: str,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    progress: AssetProgressCallback | None = None,
) -> ResolvedVoicePrompt:
    """Compatibility wrapper for current-identity managed prompt fetching."""
    return fetch_voice_prompt(
        ref,
        cache_dir=cache_dir,
        offline=offline,
        progress=progress,
    )


def list_voice_prompts(
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    dataset: str | None = None,
    variant: str | None = None,
    license: str | None = None,
    progress: AssetProgressCallback | None = None,
) -> tuple[Any, ...]:
    module = _onnxvoice()
    manager = module.OnnxVoice(cache_dir=cache_dir, offline=offline)
    return _call(
        "voice_prompt",
        lambda: manager.list_pocket_voice_prompts(
            dataset=dataset,
            variant=variant,
            license=license,
            progress=adapt_asset_progress(progress),
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

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import version

        __version__ = version("pocketsynth")
    except Exception:
        __version__ = "0.0.0"

from .api_contract import REQUEST_API_VERSION, RequestApiContract, request_api_contract
from .asset_manager import BundleAssetManager
from .asset_progress import AssetProgressCallback, AssetProgressEvent, ConsoleAssetProgress
from .assets import PocketBundle
from .bundle import BundleMetadata, BundlePaths
from .config import GenerationConfig
from .diagnostics import RuntimeDiagnostics, SynthesisTiming
from .discovery import DiscoveredBundle, DiscoveredVoice, discover_bundles, runtime_identity
from .errors import (
    AssetAccessError,
    AssetCacheError,
    AssetDownloadError,
    AssetError,
    BundleError,
    BundleLanguageError,
    BundleNotFoundError,
    CatalogUnavailableError,
    EmptyTextError,
    InvalidGenerationConfigError,
    InvalidLanguageError,
    InvalidRequestError,
    InvalidVoiceError,
    InvalidVoicePromptMetadataError,
    ModelInferenceError,
    OfflineAssetError,
    OptionalDependencyError,
    PocketSynthError,
    RuntimeCapabilityError,
    RuntimeClosedError,
    SessionCreationError,
    SynthesisError,
    SynthesisInputTooLongError,
    UnsupportedBundleError,
    UnsupportedFeatureError,
    VoicePromptChangedError,
    VoicePromptError,
)
from .runtime import PocketRuntime
from .types import (
    LinguisticToken,
    PronunciationOverride,
    RenderedChunk,
    RenderedSegment,
    RequestMeasure,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    WordTiming,
)
from .voice import PreparedVoice
from .voice_level import (
    VoiceLevelApplication,
    VoiceLevelConfig,
    VoiceLevelMode,
    VoiceLevelSource,
)
from .voice_prompts import VoicePromptInfo, inspect_voice_prompt, list_voice_prompts

__all__ = [
    "AssetAccessError",
    "AssetCacheError",
    "AssetDownloadError",
    "AssetError",
    "AssetProgressCallback",
    "AssetProgressEvent",
    "BundleAssetManager",
    "BundleError",
    "BundleLanguageError",
    "BundleMetadata",
    "BundleNotFoundError",
    "BundlePaths",
    "CatalogUnavailableError",
    "ConsoleAssetProgress",
    "DiscoveredBundle",
    "DiscoveredVoice",
    "EmptyTextError",
    "GenerationConfig",
    "InvalidGenerationConfigError",
    "InvalidLanguageError",
    "InvalidRequestError",
    "InvalidVoiceError",
    "InvalidVoicePromptMetadataError",
    "LinguisticToken",
    "ModelInferenceError",
    "OfflineAssetError",
    "OptionalDependencyError",
    "PocketBundle",
    "PocketRuntime",
    "PocketSynthError",
    "PreparedVoice",
    "PronunciationOverride",
    "RenderedChunk",
    "RenderedSegment",
    "REQUEST_API_VERSION",
    "RequestApiContract",
    "RequestMeasure",
    "RuntimeCapabilityError",
    "RuntimeClosedError",
    "RuntimeDiagnostics",
    "SessionCreationError",
    "SynthesisError",
    "SynthesisInputTooLongError",
    "SynthesisRequest",
    "SynthesisResult",
    "SynthesisSegment",
    "SynthesisTiming",
    "UnsupportedBundleError",
    "UnsupportedFeatureError",
    "VoiceLevelApplication",
    "VoiceLevelConfig",
    "VoiceLevelMode",
    "VoiceLevelSource",
    "VoicePromptChangedError",
    "VoicePromptError",
    "VoicePromptInfo",
    "discover_bundles",
    "inspect_voice_prompt",
    "list_voice_prompts",
    "runtime_identity",
    "request_api_contract",
    "WordTiming",
]

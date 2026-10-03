from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import version

        __version__ = version("pocketsynth")
    except Exception:
        __version__ = "0.0.0"

from .asset_manager import BundleAssetManager
from .asset_progress import AssetProgressCallback, AssetProgressEvent, ConsoleAssetProgress
from .assets import PocketBundle
from .bundle import BundleMetadata, BundlePaths
from .config import GenerationConfig
from .diagnostics import RuntimeDiagnostics, SynthesisTiming
from .discovery import DiscoveredBundle, DiscoveredVoice, discover_bundles, runtime_identity
from .errors import (
    BundleError,
    BundleLanguageError,
    BundleNotFoundError,
    EmptyTextError,
    InvalidGenerationConfigError,
    InvalidLanguageError,
    InvalidRequestError,
    InvalidVoiceError,
    InvalidVoicePromptMetadataError,
    ModelInferenceError,
    OptionalDependencyError,
    PocketSynthError,
    RuntimeClosedError,
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
    "AssetProgressCallback",
    "AssetProgressEvent",
    "BundleAssetManager",
    "BundleError",
    "BundleLanguageError",
    "BundleMetadata",
    "BundleNotFoundError",
    "BundlePaths",
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
    "OptionalDependencyError",
    "PocketBundle",
    "PocketRuntime",
    "PocketSynthError",
    "PreparedVoice",
    "PronunciationOverride",
    "RenderedChunk",
    "RenderedSegment",
    "RuntimeClosedError",
    "RuntimeDiagnostics",
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
    "WordTiming",
]

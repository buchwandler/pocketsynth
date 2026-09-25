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
from .convenience import synthesize, synthesize_to_wav
from .diagnostics import RuntimeDiagnostics, SynthesisTiming
from .errors import (
    BundleError,
    BundleLanguageError,
    BundleNotFoundError,
    ModelInferenceError,
    OptionalDependencyError,
    PocketSynthError,
    RuntimeClosedError,
    UnsupportedBundleError,
    VoicePromptError,
)
from .runtime import PocketRuntime
from .text_split import SentenceSplitMode
from .types import RenderedChunk, RenderedSegment, SynthesisSegment
from .voice import PreparedVoice

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
    "GenerationConfig",
    "ModelInferenceError",
    "OptionalDependencyError",
    "PocketBundle",
    "PocketRuntime",
    "PocketSynthError",
    "PreparedVoice",
    "RenderedChunk",
    "RenderedSegment",
    "RuntimeClosedError",
    "RuntimeDiagnostics",
    "SentenceSplitMode",
    "SynthesisSegment",
    "SynthesisTiming",
    "UnsupportedBundleError",
    "VoicePromptError",
    "synthesize",
    "synthesize_to_wav",
]

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
from .assets import PocketBundle
from .bundle import BundleMetadata, BundlePaths
from .config import GenerationConfig, PipelineConfig
from .pipeline import PocketPipeline, PreparedAudioUnits
from .runtime import PocketRuntime
from .types import AudioChunk, AudioMarker, AudioResult, AudioUnitDescriptor, AudioUnitResult
from .voice import PreparedVoice

from .convenience import synthesize, synthesize_to_wav
from .diagnostics import RuntimeDiagnostics, TimingDiagnostics
from .errors import (
    BundleError,
    BundleNotFoundError,
    ModelInferenceError,
    OptionalDependencyError,
    PipelineClosedError,
    PocketSynthError,
    VoiceBindingError,
    VoicePromptError,
)

__all__ = [
    "AudioMarker",
    "AudioChunk",
    "AudioResult",
    "AudioUnitDescriptor",
    "AudioUnitResult",
    "BundleAssetManager",
    "BundleError",
    "BundleMetadata",
    "BundleNotFoundError",
    "BundlePaths",
    "GenerationConfig",
    "ModelInferenceError",
    "OptionalDependencyError",
    "PipelineClosedError",
    "PipelineConfig",
    "PocketBundle",
    "PocketPipeline",
    "PocketRuntime",
    "PocketSynthError",
    "PreparedAudioUnits",
    "PreparedVoice",
    "RuntimeDiagnostics",
    "TimingDiagnostics",
    "VoiceBindingError",
    "VoicePromptError",
    "synthesize",
    "synthesize_to_wav",
]

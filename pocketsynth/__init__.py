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
from .types import AudioChunk, AudioResult, AudioUnitDescriptor, AudioUnitResult
from .voice import PreparedVoice

__all__ = [
    "__version__",
    "AudioChunk",
    "AudioResult",
    "AudioUnitDescriptor",
    "AudioUnitResult",
    "BundleAssetManager",
    "BundleMetadata",
    "BundlePaths",
    "GenerationConfig",
    "PipelineConfig",
    "PocketBundle",
    "PocketPipeline",
    "PocketRuntime",
    "PreparedAudioUnits",
    "PreparedVoice",
]

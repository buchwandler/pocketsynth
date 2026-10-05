from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

import pocketsynth

required = (
    "PocketRuntime",
    "SynthesisRequest",
    "SynthesisResult",
    "GenerationConfig",
    "RequestMeasure",
    "VoiceLevelConfig",
    "SynthesisInputTooLongError",
    "BundleAssetManager",
    "PreparedVoice",
    "VoicePromptInfo",
    "discover_bundles",
    "runtime_identity",
    "REQUEST_API_VERSION",
    "request_api_contract",
)

for name in required:
    assert getattr(pocketsynth, name) is not None, name

assert pocketsynth.REQUEST_API_VERSION == 1
assert callable(pocketsynth.PocketRuntime.from_resolved)
assert callable(pocketsynth.PocketRuntime.prepare_voice)
assert callable(pocketsynth.PocketRuntime.measure_request)
assert callable(pocketsynth.PocketRuntime.synthesize)
assert pocketsynth.request_api_contract().version == 1
assert "onnxvoice" not in sys.modules

package_file = Path(pocketsynth.__file__).resolve()
purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
assert purelib in package_file.parents, (package_file, purelib)

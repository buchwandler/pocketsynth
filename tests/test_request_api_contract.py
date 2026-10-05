from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pocketsynth import (
    REQUEST_API_VERSION,
    PocketRuntime,
    RequestApiContract,
    SynthesisRequest,
    SynthesisResult,
    request_api_contract,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RUNTIME_ERRORS = (
    "AssetError",
    "AssetDownloadError",
    "AssetAccessError",
    "AssetCacheError",
    "CatalogUnavailableError",
    "OfflineAssetError",
    "SessionCreationError",
    "RuntimeCapabilityError",
)


def test_request_api_contract_is_public() -> None:
    contract = request_api_contract()

    assert REQUEST_API_VERSION == 1
    assert isinstance(contract, RequestApiContract)
    assert contract.version == REQUEST_API_VERSION
    assert contract.request_type is SynthesisRequest
    assert contract.result_type is SynthesisResult
    assert contract.runtime_type is PocketRuntime
    assert (
        contract.synthesis_method,
        contract.measurement_method,
        contract.discovery_method,
        contract.runtime_identity_method,
    ) == ("synthesize", "measure_request", "discover_bundles", "runtime_identity")


def test_request_api_contract_declares_atomic_caller_owned_boundaries() -> None:
    contract = request_api_contract()

    assert contract.caller_owns_text_boundaries is True
    assert contract.supports_linguistic_tokens is False
    assert contract.supports_pronunciation_overrides is False
    assert contract.supports_whole_request_phonemes is False
    assert contract.supports_named_voices is True
    assert contract.supports_reference_voice is True
    assert contract.supports_managed_reference_voice is True
    assert contract.supports_word_timings is False
    assert contract.supports_voice_level is True
    assert contract.supports_request_measurement is True


def test_required_public_runtime_errors_are_exported_from_package_root() -> None:
    import pocketsynth

    for name in PUBLIC_RUNTIME_ERRORS:
        assert getattr(pocketsynth, name) is not None
        assert name in pocketsynth.__all__


def test_request_api_contract_import_is_dependency_light() -> None:
    code = """
import sys
from pocketsynth import request_api_contract
contract = request_api_contract()
assert contract.version == 1
assert 'onnxvoice' not in sys.modules
assert 'pocketsynth.convenience' not in sys.modules
assert 'pocketsynth.text_split' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

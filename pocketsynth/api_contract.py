from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUEST_API_VERSION = 1


@dataclass(frozen=True, slots=True)
class RequestApiContract:
    """Dependency-light declaration of PocketSynth's public request API."""

    version: int
    request_type: type[Any]
    result_type: type[Any]
    runtime_type: type[Any]
    synthesis_method: str
    measurement_method: str
    discovery_method: str
    runtime_identity_method: str
    caller_owns_text_boundaries: bool
    supports_linguistic_tokens: bool
    supports_pronunciation_overrides: bool
    supports_whole_request_phonemes: bool
    supports_named_voices: bool
    supports_reference_voice: bool
    supports_managed_reference_voice: bool
    supports_word_timings: bool
    supports_voice_level: bool
    supports_request_measurement: bool


def request_api_contract() -> RequestApiContract:
    """Return the PocketSynth request API declaration without opening a runtime."""
    from .runtime import PocketRuntime
    from .types import SynthesisRequest, SynthesisResult

    return RequestApiContract(
        version=REQUEST_API_VERSION,
        request_type=SynthesisRequest,
        result_type=SynthesisResult,
        runtime_type=PocketRuntime,
        synthesis_method="synthesize",
        measurement_method="measure_request",
        discovery_method="discover_bundles",
        runtime_identity_method="runtime_identity",
        caller_owns_text_boundaries=True,
        supports_linguistic_tokens=False,
        supports_pronunciation_overrides=False,
        supports_whole_request_phonemes=False,
        supports_named_voices=True,
        supports_reference_voice=True,
        supports_managed_reference_voice=True,
        supports_word_timings=False,
        supports_voice_level=True,
        supports_request_measurement=True,
    )

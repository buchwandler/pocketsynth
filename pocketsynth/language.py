from __future__ import annotations

from .bundle import BundleMetadata

_LANGUAGE_MAP = {
    "english_2026-04": "en",
    "english": "en",
    "en": "en",
    "french_24l": "fr",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "german_24l": "de",
    "de": "de",
    "italian": "it",
    "italian_24l": "it",
    "it": "it",
    "portuguese": "pt",
    "portuguese_24l": "pt",
    "pt": "pt",
    "spanish": "es",
    "spanish_24l": "es",
    "es": "es",
}


def normalize_language(language: str) -> str:
    value = language.strip().casefold().replace("_", "-")
    return _LANGUAGE_MAP.get(
        value, _LANGUAGE_MAP.get(value.split("-", 1)[0], value.split("-", 1)[0])
    )


def bundle_language(metadata: BundleMetadata) -> str:
    """Return the normalized language declared by a Pocket bundle."""
    return normalize_language(metadata.language)

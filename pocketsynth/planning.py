from __future__ import annotations

from typing import Literal

from utterplan import PlannerConfig, normalize_language

from .bundle import BundleMetadata
from .config import PipelineConfig

_LANGUAGE_MAP = {
    "english_2026-04": "en",
    "french_24l": "fr",
    "german": "de",
    "german_24l": "de",
    "italian": "it",
    "italian_24l": "it",
    "portuguese": "pt",
    "portuguese_24l": "pt",
    "spanish": "es",
    "spanish_24l": "es",
}


def planner_language(metadata: BundleMetadata) -> str:
    return normalize_language(_LANGUAGE_MAP.get(metadata.language, metadata.language.split("_", 1)[0]))


def planner_config_from_pocketsynth(
    config: PipelineConfig,
    metadata: BundleMetadata,
    *,
    unit: Literal["paragraph", "sentence"] | None = None,
) -> PlannerConfig:
    language = config.language or planner_language(metadata)
    return PlannerConfig(
        language=normalize_language(language, dict(config.language_aliases)),
        document_format=config.document_format,
        text_preparation=config.text_preparation,
        unit=unit or config.unit,
        pauses=config.pauses,
        linguistics=config.linguistics,
        ssmd=config.ssmd,
        overlap_mode=config.overlap_mode,
        language_aliases=dict(config.language_aliases),
        diagnostics=config.planner_diagnostics,
    )

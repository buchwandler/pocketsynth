from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from utterplan import LinguisticsConfig, PauseConfig, SSMDConfig

from .bundle import Precision


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 0.7
    lsd_steps: int = 1
    max_frames: int | None = None
    frames_after_eos: int | None = None
    normalize_audio: bool = False
    volume: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.lsd_steps < 1:
            raise ValueError("lsd_steps must be >= 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if self.frames_after_eos is not None and self.frames_after_eos < 0:
            raise ValueError("frames_after_eos must be >= 0")
        if self.volume < 0:
            raise ValueError("volume must be >= 0")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    bundle_dir: Path
    precision: Precision = "int8"
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    providers: Any | None = None
    provider_options: Mapping[str, Any] | None = None
    session_options: Any | None = None
    language: str | None = None
    document_format: Literal["plain", "ssmd"] = "plain"
    text_preparation: Literal["identity", "spokenform"] = "spokenform"
    unit: Literal["paragraph", "sentence"] = "sentence"
    pauses: PauseConfig = field(default_factory=PauseConfig)
    linguistics: LinguisticsConfig = field(default_factory=LinguisticsConfig)
    ssmd: SSMDConfig = field(default_factory=SSMDConfig)
    overlap_mode: Literal["snap", "strict"] = "snap"
    language_aliases: Mapping[str, str] = field(default_factory=dict)
    planner_diagnostics: bool = True
    directive_policy: Literal["error", "warn", "ignore"] = "error"
    language_policy: Literal["strict", "allow"] = "strict"
    retain_unit_audio: bool = False
    return_diagnostics: bool = True

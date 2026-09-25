from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .errors import InvalidGenerationConfigError


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 0.7
    lsd_steps: int = 1
    max_frames: int | None = None
    frames_after_eos: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, Real)
            or not math.isfinite(float(self.temperature))
            or not 0.0 <= self.temperature <= 2.0
        ):
            raise InvalidGenerationConfigError("temperature must be finite and between 0 and 2")
        if (
            isinstance(self.lsd_steps, bool)
            or not isinstance(self.lsd_steps, int)
            or self.lsd_steps < 1
        ):
            raise InvalidGenerationConfigError("lsd_steps must be an integer >= 1")
        if self.max_frames is not None and (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, int)
            or self.max_frames < 1
        ):
            raise InvalidGenerationConfigError("max_frames must be an integer >= 1 or None")
        if self.frames_after_eos is not None and (
            isinstance(self.frames_after_eos, bool)
            or not isinstance(self.frames_after_eos, int)
            or self.frames_after_eos < 0
        ):
            raise InvalidGenerationConfigError("frames_after_eos must be an integer >= 0 or None")

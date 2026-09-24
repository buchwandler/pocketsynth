from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 0.7
    lsd_steps: int = 1
    max_frames: int | None = None
    frames_after_eos: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.lsd_steps < 1:
            raise ValueError("lsd_steps must be >= 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if self.frames_after_eos is not None and self.frames_after_eos < 0:
            raise ValueError("frames_after_eos must be >= 0")

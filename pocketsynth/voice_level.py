from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from .errors import InvalidGenerationConfigError, UnsupportedBundleError

VoiceLevelMode = Literal["off", "calibrated"]
VoiceLevelSource = Literal["off", "override", "catalog", "missing_calibration"]


@dataclass(frozen=True, slots=True)
class VoiceLevelConfig:
    """Static voice-level calibration settings for one synthesis request."""

    mode: VoiceLevelMode = "off"
    gain_db: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"off", "calibrated"}:
            raise InvalidGenerationConfigError("voice level mode must be 'off' or 'calibrated'")
        if self.gain_db is not None and (
            isinstance(self.gain_db, bool)
            or not isinstance(self.gain_db, Real)
            or not math.isfinite(float(self.gain_db))
        ):
            raise InvalidGenerationConfigError("voice level gain_db must be finite or None")


@dataclass(frozen=True, slots=True)
class VoiceLevelApplication:
    """Structured record of the static gain policy applied to a rendered result."""

    mode: VoiceLevelMode
    gain_db: float
    source: VoiceLevelSource
    applied: bool


def apply_voice_level_calibration(
    audio: np.ndarray,
    config: VoiceLevelConfig,
    *,
    identity: Mapping[str, str | int | None] | None,
    catalog: Mapping[str, object],
) -> tuple[np.ndarray, VoiceLevelApplication]:
    """Apply only a fixed catalog or explicit gain, never waveform analysis."""
    source: VoiceLevelSource = "off"
    gain_db = 0.0
    if config.gain_db is not None:
        gain_db = float(config.gain_db)
        source = "override"
    elif config.mode == "calibrated":
        voice_name = identity.get("name") if identity is not None else None
        if (
            identity is not None
            and identity.get("kind") == "predefined"
            and isinstance(voice_name, str)
        ):
            catalog_gain = catalog.get(voice_name)
            if catalog_gain is not None:
                if (
                    isinstance(catalog_gain, bool)
                    or not isinstance(catalog_gain, Real)
                    or not math.isfinite(float(catalog_gain))
                ):
                    raise UnsupportedBundleError(
                        f"Invalid voice_level_calibration gain for {voice_name!r}"
                    )
                gain_db = float(catalog_gain)
                source = "catalog"
            else:
                source = "missing_calibration"
        else:
            source = "missing_calibration"

    scale = np.float32(10.0 ** (gain_db / 20.0))
    calibrated_audio = np.asarray(audio, dtype=np.float32) * scale
    application = VoiceLevelApplication(
        mode=config.mode,
        gain_db=gain_db,
        source=source,
        applied=gain_db != 0.0,
    )
    return calibrated_audio, application

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    bundle_id: str | None = None
    bundle_path: str | None = None
    sample_rate: int | None = None
    precision: str | None = None
    providers_requested: tuple[str, ...] = ()
    providers_active: tuple[str, ...] = ()
    sessions: tuple[str, ...] = ()
    source_revision: str | None = None


@dataclass(frozen=True, slots=True)
class SynthesisTiming:
    frontend_ms: float | None = None
    inference_ms: float | None = None
    total_ms: float | None = None

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    bundle_id: str | None = None
    bundle_path: str | None = None
    sample_rate: int | None = None
    precision: str | None = None
    providers_requested: tuple[str, ...] = ()
    providers_active: tuple[str, ...] = ()
    sessions: tuple[str, ...] = ()
    plan_id: str | None = None
    utterplan_producer: dict[str, Any] | None = None
    utterplan_schema_version: int | None = None
    source_revision: str | None = None


@dataclass(frozen=True, slots=True)
class TimingDiagnostics:
    planning_ms: float | None = None
    frontend_ms: float | None = None
    inference_ms: float | None = None
    composition_ms: float | None = None
    total_ms: float | None = None

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TextIO

ProgressPhase = Literal["catalog", "download", "verify", "install"]
AssetProgressCallback = Callable[["AssetProgressEvent"], None]


@dataclass(frozen=True, slots=True)
class AssetProgressEvent:
    """Stable PocketSynth view of managed asset progress."""

    phase: ProgressPhase
    status: str
    ref: str | None = None
    artifact: str | None = None
    completed: int | None = None
    total: int | None = None
    message: str | None = None
    role: str | None = None
    target: str | None = None

    @property
    def fraction(self) -> float | None:
        if self.completed is None or self.total in (None, 0):
            return None
        return min(1.0, max(0.0, self.completed / self.total))


def _phase_for(status: str) -> ProgressPhase:
    if status.startswith("catalog_"):
        return "catalog"
    if status.startswith("download_"):
        return "download"
    if status.startswith("verify_"):
        return "verify"
    return "install"


def _to_event(value: Any) -> AssetProgressEvent:
    if isinstance(value, AssetProgressEvent):
        return value
    status = str(getattr(value, "phase", "progress"))
    return AssetProgressEvent(
        phase=_phase_for(status),
        status=status,
        ref=getattr(value, "ref", None),
        artifact=getattr(value, "artifact", None),
        completed=getattr(value, "completed", None),
        total=getattr(value, "total", None),
        message=getattr(value, "message", None),
        role=getattr(value, "role", None),
        target=getattr(value, "target", None),
    )


def adapt_asset_progress(callback: AssetProgressCallback | None) -> Callable[[Any], None] | None:
    """Adapt an application callback to the upstream OnnxVoice event shape."""
    if callback is None:
        return None

    def receive(value: Any) -> None:
        callback(_to_event(value))

    return receive


class ConsoleAssetProgress:
    """Render managed asset progress as concise human-readable lines."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout

    def __call__(self, event: AssetProgressEvent) -> None:
        details = [event.status]
        if event.artifact:
            details.append(event.artifact)
        if event.completed is not None and event.total:
            details.append(f"{event.completed}/{event.total}")
        if event.message:
            details.append(event.message)
        print(f"[{event.phase}] {' '.join(details)}", file=self.stream)

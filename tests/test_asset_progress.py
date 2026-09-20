from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

from pocketsynth.asset_progress import (
    AssetProgressEvent,
    ConsoleAssetProgress,
    adapt_asset_progress,
)


@dataclass
class UpstreamProgress:
    phase: str
    ref: str
    artifact: str | None = None
    completed: int | None = None
    total: int | None = None
    message: str | None = None
    role: str | None = None
    target: str | None = None


def test_progress_adapter_maps_upstream_fields_and_phase() -> None:
    received: list[AssetProgressEvent] = []
    callback = adapt_asset_progress(received.append)

    assert callback is not None
    callback(UpstreamProgress("download_progress", "pocket:english", "model.onnx", 3, 10))

    assert received == [
        AssetProgressEvent(
            phase="download",
            status="download_progress",
            ref="pocket:english",
            artifact="model.onnx",
            completed=3,
            total=10,
        )
    ]
    assert received[0].fraction == 0.3


def test_console_progress_is_concise() -> None:
    stream = StringIO()
    render = ConsoleAssetProgress(stream)
    render(AssetProgressEvent("install", "artifact_installed", artifact="model.onnx", completed=1, total=2))

    assert stream.getvalue() == "[install] artifact_installed model.onnx 1/2\n"

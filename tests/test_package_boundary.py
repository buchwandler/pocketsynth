from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "pocketsynth"
FORBIDDEN_IMPORTS = {"utterplan", "ssmd", "audiocompose"}
FORBIDDEN_TERMS = (
    "utterplan",
    "ssmd",
    "audiocompose",
    "utteranceplan",
    "utteranceplanner",
    "pocketpipeline",
    "pipelineconfig",
    "preparedaudiounits",
    "audiojob",
    "audiomarker",
    "audiounitdescriptor",
    "audiounitresult",
    "planner_language",
    "planner_diagnostics",
    "planning_ms",
    "composition_ms",
    "composer",
    "compositionresult",
    "audioresult",
    "voicebindingerror",
    "unsupportedplandirectiveerror",
    "unsupportedplanlanguageerror",
    "pipelineclosederror",
    "plan_adapter",
    "planning.py",
    "audio_job",
    "planner_config",
    "plan_id",
    "unit_id",
    "pause_before",
    "pause_after",
    "marker_ids",
    "language_policy",
    "normalize_audio",
    "retain_unit_audio",
)


def test_runtime_package_has_no_neighboring_layer_imports() -> None:
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules = {
            node.name.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for node in node.names
        }
        imported_modules.update(
            node.module.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert imported_modules.isdisjoint(FORBIDDEN_IMPORTS), path


def test_runtime_package_has_no_removed_architecture_terminology() -> None:
    for path in PACKAGE.rglob("*.py"):
        source = path.read_text(encoding="utf-8").casefold()
        for term in FORBIDDEN_TERMS:
            assert re.search(rf"\b{re.escape(term)}\b", source) is None, (path, term)


def test_project_metadata_has_only_supported_engine_dependencies() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8").casefold()
    assert all(name not in metadata for name in FORBIDDEN_IMPORTS)
    assert '"onnxvoice>=0.1.12,<0.2"' in metadata
    assert '"onnxvoice[cpu,pocket]>=0.1.12,<0.2"' in metadata
    assert '"onnxvoice[gpu,pocket]>=0.1.12,<0.2"' in metadata

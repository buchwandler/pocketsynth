from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_example_runner_lists_first_wav_paths() -> None:
    result = subprocess.run(
        [sys.executable, "examples/run_all.py", "--list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "quickstart.py\tmanaged" in result.stdout
    assert "local_bundle.py\tlocal" in result.stdout


def test_quickstart_help_is_available_without_runtime_assets() -> None:
    result = subprocess.run(
        [sys.executable, "examples/quickstart.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--voice" in result.stdout
    assert "--output" in result.stdout

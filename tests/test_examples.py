from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from examples.clone_all_kyutai_voices import main as clone_all_main
from examples.clone_all_kyutai_voices import safe_output_name
from examples.kyutai_voice import main as kyutai_voice_main

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
    assert "predefined_voice.py\tmanaged" in result.stdout
    assert "kyutai_voice.py\tmanaged" in result.stdout
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


def test_long_text_example_help_is_available_without_runtime_assets() -> None:
    result = subprocess.run(
        [sys.executable, "examples/long_text.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    help_text = " ".join(result.stdout.split())
    assert "lightweight sentence segmentation" in help_text
    assert "sentence_split='none' bypass" in help_text


def test_kyutai_voice_example_help_is_available_without_runtime_assets() -> None:
    result = subprocess.run(
        [sys.executable, "examples/kyutai_voice.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    help_text = " ".join(result.stdout.split())
    assert "--voice VOICE" in help_text
    assert "--offline" in result.stdout


def test_clone_all_example_help_lists_requested_options() -> None:
    result = subprocess.run(
        [sys.executable, "examples/clone_all_kyutai_voices.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for option in (
        "--bundle",
        "--text",
        "--output-dir",
        "--dataset",
        "--variant",
        "--limit",
        "--dry-run",
        "--offline",
        "--exclude-noncommercial",
    ):
        assert option in result.stdout


def test_kyutai_voice_example_uses_managed_ref_without_manual_download(tmp_path: Path) -> None:
    runtime = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = runtime
    context.__exit__.return_value = False
    output = tmp_path / "sample.wav"

    with patch(
        "examples.kyutai_voice.PocketRuntime.from_pretrained", return_value=context
    ) as factory:
        status = kyutai_voice_main(
            [
                "--bundle",
                "test-bundle",
                "--voice",
                "kyutai-tts-voices:alba-mackenna/casual",
                "--output",
                str(output),
                "--offline",
            ]
        )

    assert status == 0
    factory.assert_called_once_with("test-bundle", cache_dir=None, offline=True)
    runtime.prepare_voice.assert_called_once_with("kyutai-tts-voices:alba-mackenna/casual")
    runtime.synthesize_text.assert_called_once()
    runtime.synthesize_text.return_value.save_wav.assert_called_once_with(output)


def test_clone_all_continues_after_errors_and_writes_manifest(tmp_path: Path) -> None:
    prompts = (
        SimpleNamespace(
            id="alba-mackenna/casual",
            ref="kyutai-tts-voices:alba-mackenna/casual",
            source_path="alba-mackenna/casual.wav",
            license="CC-BY-4.0",
            variant="casual",
        ),
        SimpleNamespace(
            id="vctk/speaker 2",
            ref="kyutai-tts-voices:vctk/speaker-2",
            source_path="vctk/speaker-2.wav",
            license="CC-BY-4.0",
            variant="casual",
        ),
        SimpleNamespace(
            id="donation/nc",
            ref="kyutai-tts-voices:donation/nc",
            source_path="donation/nc.wav",
            license="CC-BY-NC-4.0",
            variant="casual",
        ),
    )
    catalog = MagicMock()
    catalog.list_pocket_voice_prompts.return_value = prompts
    runtime = MagicMock()
    runtime.__enter__.return_value = runtime
    runtime.__exit__.return_value = False
    prepared: list[str] = []
    audio = MagicMock()
    audio.save_wav.side_effect = lambda path: Path(path).write_bytes(b"fake wav")
    runtime.synthesize_text.return_value = audio

    def prepare(ref: str) -> str:
        prepared.append(ref)
        if ref == prompts[1].ref:
            raise RuntimeError("broken sample")
        return ref

    runtime.prepare_voice.side_effect = prepare
    output_dir = tmp_path / "voices"

    with (
        patch("examples.clone_all_kyutai_voices.OnnxVoice", return_value=catalog),
        patch(
            "examples.clone_all_kyutai_voices.PocketRuntime.from_pretrained", return_value=runtime
        ) as factory,
    ):
        status = clone_all_main(
            [
                "--bundle",
                "test-bundle",
                "--text",
                "sample text",
                "--output-dir",
                str(output_dir),
                "--dataset",
                "alba-mackenna",
                "--variant",
                "casual",
                "--limit",
                "2",
                "--offline",
                "--exclude-noncommercial",
            ]
        )

    assert status == 1
    catalog.list_pocket_voice_prompts.assert_called_once_with(
        dataset="alba-mackenna", variant="casual"
    )
    factory.assert_called_once_with("test-bundle", cache_dir=None, offline=True)
    assert prepared == [prompts[0].ref, prompts[1].ref]
    assert (output_dir / "alba-mackenna__casual.wav").is_file()
    assert not (output_dir / "vctk__speaker__2.wav").exists()
    with (output_dir / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == ["ok", "error"]
    assert rows[1]["error"] == "RuntimeError: broken sample"
    assert rows[1]["output"].endswith("vctk__speaker__2.wav")


def test_clone_all_dry_run_discovers_every_prompt_without_opening_runtime(tmp_path: Path) -> None:
    prompts = tuple(
        SimpleNamespace(
            id=f"speaker/{index}",
            ref=f"kyutai-tts-voices:speaker/{index}",
            source_path=f"speaker/{index}.wav",
            license="CC-BY-4.0",
            variant="casual",
        )
        for index in range(3)
    )
    catalog = MagicMock()
    catalog.list_pocket_voice_prompts.return_value = prompts
    output_dir = tmp_path / "dry-run"

    with (
        patch("examples.clone_all_kyutai_voices.OnnxVoice", return_value=catalog),
        patch("examples.clone_all_kyutai_voices.PocketRuntime.from_pretrained") as factory,
    ):
        status = clone_all_main(["--output-dir", str(output_dir), "--dry-run"])

    assert status == 0
    factory.assert_not_called()
    with (output_dir / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(prompts)
    assert all(row["status"] == "dry-run" for row in rows)
    assert safe_output_name("speaker/../unsafe name") == "speaker__..__unsafe__name.wav"

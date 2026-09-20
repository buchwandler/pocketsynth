from __future__ import annotations

from unittest.mock import MagicMock, patch

from pocketsynth.__main__ import main


def test_synthesize_cli_wires_generation_and_cache_options(tmp_path) -> None:
    pipeline = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = pipeline
    context.__exit__.return_value = False
    with patch(
        "pocketsynth.__main__.PocketPipeline.from_pretrained", return_value=context
    ) as factory:
        assert (
            main(
                [
                    "synthesize",
                    "--bundle",
                    "english_2026-04",
                    "--voice",
                    "voice.wav",
                    "--output",
                    str(tmp_path / "out.wav"),
                    "--temperature",
                    "0.4",
                    "--lsd-steps",
                    "3",
                    "--max-frames",
                    "50",
                    "--offline",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--provider",
                    "CPUExecutionProvider",
                    "Hello",
                ]
            )
            == 0
        )

    kwargs = factory.call_args.kwargs
    assert kwargs["offline"] is True
    assert kwargs["cache_dir"] == tmp_path / "cache"
    assert kwargs["providers"] == "CPUExecutionProvider"
    assert kwargs["generation"].temperature == 0.4
    assert kwargs["generation"].lsd_steps == 3
    assert kwargs["generation"].max_frames == 50
    context.set_default_voice.assert_called_once()
    context.run.assert_called_once_with("Hello")


def test_check_reports_available_provider(capsys) -> None:
    assert main(["check", "--provider", "CPUExecutionProvider"]) == 0
    output = capsys.readouterr().out
    assert "OnnxVoice:" in output
    assert "ORT providers:" in output

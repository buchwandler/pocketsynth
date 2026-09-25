from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pocketsynth.__main__ import main


@pytest.mark.parametrize("voice", ["alba", "voice.wav"])
@pytest.mark.parametrize(
    ("sentence_split_args", "expected_sentence_split"),
    [
        ([], "none"),
        (["--sentence-split", "phrasplit"], "phrasplit"),
        (["--sentence-split", "none"], "none"),
    ],
)
def test_synthesize_cli_wires_generation_and_cache_options(
    tmp_path, voice, sentence_split_args, expected_sentence_split
) -> None:
    runtime = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = runtime
    result = MagicMock(sample_rate=24_000, duration_seconds=1.0)
    context.__exit__.return_value = False
    output = tmp_path / "out.wav"

    with (
        patch(
            "pocketsynth.__main__.PocketRuntime.from_pretrained", return_value=context
        ) as factory,
        patch("pocketsynth.__main__.synthesize_with_runtime", return_value=result) as render,
    ):
        assert (
            main(
                [
                    "synthesize",
                    "--bundle",
                    "english_2026-04",
                    "--voice",
                    voice,
                    "--output",
                    str(output),
                    "--temperature",
                    "0.4",
                    "--lsd-steps",
                    "3",
                    "--max-frames",
                    "50",
                    "--offline",
                    "--refresh-catalog",
                    "--force-download",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--provider",
                    "CPUExecutionProvider",
                    *sentence_split_args,
                    "Hello",
                ]
            )
            == 0
        )

    factory.assert_called_once_with(
        "english_2026-04",
        precision="int8",
        providers="CPUExecutionProvider",
        cache_dir=tmp_path / "cache",
        offline=True,
        refresh_catalog=True,
        force_download=True,
    )
    render.assert_called_once()
    args, kwargs = render.call_args
    assert args == (runtime, "Hello")
    assert kwargs["voice"] == voice
    assert kwargs["sentence_split"] == expected_sentence_split
    generation = kwargs["generation"]
    assert generation.temperature == 0.4
    assert generation.lsd_steps == 3
    assert generation.max_frames == 50
    result.save_wav.assert_called_once_with(output)


def test_check_reports_available_provider(capsys) -> None:
    assert main(["check", "--provider", "CPUExecutionProvider"]) == 0
    output = capsys.readouterr().out
    assert "OnnxVoice:" in output
    assert "ORT providers:" in output


@pytest.mark.parametrize(
    ("voice", "expected_status", "expected_text"),
    [
        ("alba", 0, "Predefined voice: OK (alba)"),
        ("other", 1, "available voices: alba"),
    ],
)
def test_check_validates_predefined_names_from_catalog(
    voice: str, expected_status: int, expected_text: str, capsys
) -> None:
    manager = MagicMock()
    manager.catalog.resolve.return_value.metadata = {"predefined_voice_names": ["alba"]}

    with (
        patch("onnxvoice.OnnxVoice", return_value=manager),
        patch("onnxvoice.available_providers", return_value=["CPUExecutionProvider"]),
        patch("pocketsynth.__main__._read_pcm_wav") as read_wav,
    ):
        status = main(["check", "--bundle", "english_2026-04", "--voice", voice])

    output = capsys.readouterr().out
    assert status == expected_status
    assert expected_text in output
    manager.catalog.resolve.assert_called_once_with("pocket:english_2026-04", quality="int8")
    manager.install.assert_not_called()
    read_wav.assert_not_called()


def test_check_validates_predefined_name_from_local_bundle(tmp_path, capsys) -> None:
    paths = MagicMock()
    paths.root = tmp_path
    paths.metadata.predefined_voices = ("alba",)

    with (
        patch("pocketsynth.__main__.BundlePaths.from_directory", return_value=paths),
        patch("onnxvoice.available_providers", return_value=["CPUExecutionProvider"]),
        patch("pocketsynth.__main__._read_pcm_wav") as read_wav,
    ):
        status = main(["check", "--bundle-dir", str(tmp_path), "--voice", "alba"])

    assert status == 0
    assert "Predefined voice: OK (alba)" in capsys.readouterr().out
    read_wav.assert_not_called()


def test_check_keeps_local_wav_path_string(tmp_path, capsys) -> None:
    wav_path = str(tmp_path / "reference.wav")
    with (
        patch("onnxvoice.available_providers", return_value=["CPUExecutionProvider"]),
        patch(
            "pocketsynth.__main__._read_pcm_wav",
            return_value=(object(), 24000),
        ) as read_wav,
    ):
        status = main(["check", "--voice", wav_path])

    assert status == 0
    assert "Voice WAV: OK (mono PCM16, sample rate 24000)" in capsys.readouterr().out
    read_wav.assert_called_once_with(wav_path)


def test_synthesize_cli_documents_sentence_split_modes(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["synthesize", "--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--sentence-split {phrasplit,none}" in help_text
    assert "Phrasplit's lightweight regex backend (no spaCy)" in help_text
    assert "none" in help_text

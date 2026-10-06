from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pocketsynth import DiscoveredBundle, VoicePromptInfo
from pocketsynth.__main__ import main
from pocketsynth.errors import VoicePromptError


def voice_prompt(ref: str) -> VoicePromptInfo:
    return VoicePromptInfo(
        ref=ref,
        source_repository="kyutai-tts-voices",
        source_revision="catalog-revision",
        source_path="speaker/prompt.wav",
        size=42,
        sha256="a" * 64,
        license="cc-by-4.0",
        dataset="alba-mackenna",
        variant="casual",
    )


def discovered_bundle() -> DiscoveredBundle:
    return DiscoveredBundle(
        id="english_2026-04",
        ref="pocket:english_2026-04",
        display_name="English",
        aliases=(),
        language="en",
        sample_rate=24_000,
        precisions=("int8", "fp32"),
        predefined_voices=("alba",),
        default_voice="alba",
        source_revision="bundle-revision",
        max_tokens=512,
    )


@pytest.mark.parametrize("voice", ["alba", "voice.wav", "kyutai-tts-voices:alba-mackenna/casual"])
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


def test_synthesize_cli_leaves_temperature_to_runtime_by_default(tmp_path) -> None:
    runtime = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = runtime
    context.__exit__.return_value = False
    result = MagicMock(sample_rate=24_000, duration_seconds=1.0)

    with (
        patch("pocketsynth.__main__.PocketRuntime.from_pretrained", return_value=context),
        patch("pocketsynth.__main__.synthesize_with_runtime", return_value=result) as render,
    ):
        assert (
            main(
                [
                    "synthesize",
                    "--bundle",
                    "english_2026-04",
                    "--voice",
                    "alba",
                    "--output",
                    str(tmp_path / "out.wav"),
                    "Hello",
                ]
            )
            == 0
        )

    assert render.call_args.kwargs["generation"].temperature is None


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
    with (
        patch(
            "pocketsynth.__main__.discover_bundles", return_value=(discovered_bundle(),)
        ) as discover,
        patch("onnxvoice.available_providers", return_value=["CPUExecutionProvider"]),
        patch("pocketsynth.__main__._read_pcm_wav") as read_wav,
    ):
        status = main(["check", "--bundle", "english_2026-04", "--voice", voice])

    output = capsys.readouterr().out
    assert status == expected_status
    assert expected_text in output
    discover.assert_called_once_with(cache_dir=None, offline=False)
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
    assert "Voice WAV: OK (sample rate 24000 Hz)" in capsys.readouterr().out
    read_wav.assert_called_once_with(wav_path)


def test_synthesize_cli_documents_sentence_split_modes(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["synthesize", "--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--sentence-split {phrasplit,none}" in help_text
    assert "Phrasplit's lightweight regex backend (no spaCy)" in help_text
    assert "none" in help_text


def test_voices_list_filters_catalog_without_fetching_audio(tmp_path, capsys) -> None:
    ref = "kyutai-tts-voices:alba-mackenna/casual"
    prompt = voice_prompt(ref)
    cache_dir = tmp_path / "cache"

    with patch("pocketsynth.__main__.list_voice_prompts", return_value=(prompt,)) as listing:
        status = main(
            [
                "voices",
                "list",
                "--cache-dir",
                str(cache_dir),
                "--offline",
                "--dataset",
                "alba-mackenna",
                "--variant",
                "casual",
                "--license",
                "cc-by-4.0",
            ]
        )

    assert status == 0
    output = capsys.readouterr().out
    assert "REF\tDATASET\tVARIANT\tLICENSE\tSOURCE_PATH" in output
    assert f"{ref}\talba-mackenna\tcasual\tcc-by-4.0\tspeaker/prompt.wav" in output
    listing.assert_called_once_with(
        cache_dir=cache_dir,
        offline=True,
        dataset="alba-mackenna",
        variant="casual",
        license="cc-by-4.0",
    )


def test_check_inspects_managed_voice_without_fetching_audio(capsys) -> None:
    ref = "kyutai-tts-voices:alba-mackenna/casual"
    prompt = voice_prompt(ref)

    with (
        patch("pocketsynth.__main__.inspect_voice_prompt", return_value=prompt) as inspect,
        patch("onnxvoice.available_providers", return_value=["CPUExecutionProvider"]),
        patch("pocketsynth.__main__._read_pcm_wav") as read_wav,
        patch("pocketsynth.__main__.PocketRuntime.from_pretrained") as factory,
    ):
        status = main(["check", "--offline", "--voice", ref])

    assert status == 0
    assert f"Managed voice prompt: OK ({ref})" in capsys.readouterr().out
    inspect.assert_called_once_with(ref, cache_dir=None, offline=True)
    read_wav.assert_not_called()
    factory.assert_not_called()


def test_synthesize_cli_reports_invalid_managed_voice_cleanly(tmp_path, capsys) -> None:
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    context.__exit__.return_value = False
    ref = "kyutai-tts-voices:missing"

    with (
        patch("pocketsynth.__main__.PocketRuntime.from_pretrained", return_value=context),
        patch(
            "pocketsynth.__main__.synthesize_with_runtime",
            side_effect=VoicePromptError("Unknown voice prompt"),
        ),
    ):
        status = main(
            [
                "synthesize",
                "--bundle",
                "english_2026-04",
                "--offline",
                "--voice",
                ref,
                "--output",
                str(tmp_path / "out.wav"),
                "Hello",
            ]
        )

    assert status == 1
    assert "FAIL synthesis: Unknown voice prompt" in capsys.readouterr().err

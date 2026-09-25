from __future__ import annotations

from typing import cast

import phrasplit
import pytest

import pocketsynth
from pocketsynth.text_split import SentenceSplitMode, split_text_for_synthesis


def test_none_mode_returns_original_text_without_calling_phrasplit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> list[str]:
        raise AssertionError("Phrasplit must not be called in none mode")

    monkeypatch.setattr(phrasplit, "split_sentences", fail_if_called)
    text = " First sentence.\n\nSecond sentence. "

    assert split_text_for_synthesis(text, language="en", mode="none") == (text,)


def test_phrasplit_mode_forces_simple_backend_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, bool]] = []

    def fake_split_sentences(
        text: str,
        *,
        language: str,
        use_spacy: bool,
    ) -> list[str]:
        calls.append((text, language, use_spacy))
        return ["First.", "", "Second.", "   ", "Third."]

    monkeypatch.setattr(phrasplit, "split_sentences", fake_split_sentences)

    result = split_text_for_synthesis(
        "First. Second. Third.",
        language="de",
        mode="phrasplit",
    )

    assert result == ("First.", "Second.", "Third.")
    assert calls == [("First. Second. Third.", "de", False)]


def test_phrasplit_handles_english_abbreviations() -> None:
    text = "Dr. Smith arrived at 9 a.m. He sat down. Then he spoke."

    assert split_text_for_synthesis(text, language="en", mode="phrasplit") == (
        "Dr. Smith arrived at 9 a.m.",
        "He sat down.",
        "Then he spoke.",
    )


def test_phrasplit_handles_german_abbreviations() -> None:
    text = "Dr. Müller kommt heute. Danach beginnt die Sitzung."

    assert split_text_for_synthesis(text, language="de", mode="phrasplit") == (
        "Dr. Müller kommt heute.",
        "Danach beginnt die Sitzung.",
    )


def test_invalid_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported sentence split mode"):
        split_text_for_synthesis(
            "A sentence.",
            language="en",
            mode=cast(SentenceSplitMode, "automatic"),
        )


def test_sentence_split_mode_is_limited_to_the_explicit_split_module() -> None:
    assert "SentenceSplitMode" not in pocketsynth.__all__
    assert not hasattr(pocketsynth, "SentenceSplitMode")

from __future__ import annotations

from typing import Literal

SentenceSplitMode = Literal["phrasplit", "none"]


def split_text_for_synthesis(
    text: str,
    *,
    language: str,
    mode: SentenceSplitMode,
) -> tuple[str, ...]:
    if mode == "none":
        return (text,)

    if mode != "phrasplit":
        raise ValueError(f"unsupported sentence split mode: {mode!r}")

    from phrasplit import split_sentences

    return tuple(
        sentence
        for sentence in split_sentences(
            text,
            language=language,
            use_spacy=False,
        )
        if sentence.strip()
    )

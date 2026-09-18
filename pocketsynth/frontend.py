from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bundle import BundleMetadata
from .errors import OptionalDependencyError

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_SOFT_END = re.compile(r"(?<=[,;:])\s+")


class PocketFrontend:
    """Pocket text preparation and SentencePiece encoding.

    UtterPlan owns semantic sentence/paragraph boundaries. This frontend only performs
    Pocket-specific text normalization and model token-limit subdivision.
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        metadata: BundleMetadata,
        *,
        processor: Any | None = None,
    ) -> None:
        self.metadata = metadata
        if processor is None:
            try:
                import sentencepiece as spm
            except ModuleNotFoundError as exc:
                raise OptionalDependencyError(
                    "Pocket tokenization requires sentencepiece. Install pocketsynth."
                ) from exc
            processor = spm.SentencePieceProcessor()
            if not processor.Load(str(tokenizer_path)):
                raise ValueError(f"Could not load SentencePiece model: {tokenizer_path}")
        self.processor = processor

    def prepare_text(self, text: str) -> str:
        value = " ".join(text.strip().split())
        if self.metadata.remove_semicolons:
            value = value.replace(";", ",")
        if self.metadata.pad_with_spaces_for_short_inputs and value:
            value = f" {value} "
        return value

    def encode(self, text: str) -> tuple[int, ...]:
        prepared = self.prepare_text(text)
        if not prepared:
            return ()
        values = self.processor.EncodeAsIds(prepared)
        return tuple(int(value) for value in values)

    def split_for_model(self, text: str) -> tuple[str, ...]:
        prepared = self.prepare_text(text)
        if not prepared:
            return ()
        limit = self.metadata.max_token_per_chunk
        if len(self.encode(prepared)) <= limit:
            return (prepared,)

        chunks = self._pack(_SENTENCE_END.split(prepared), limit)
        result: list[str] = []
        for chunk in chunks:
            if len(self.encode(chunk)) <= limit:
                result.append(chunk)
                continue
            for soft in self._pack(_SOFT_END.split(chunk), limit):
                if len(self.encode(soft)) <= limit:
                    result.append(soft)
                else:
                    result.extend(self._split_words(soft, limit))
        return tuple(part for part in result if part.strip())

    def _pack(self, parts: Sequence[str], limit: int) -> list[str]:
        result: list[str] = []
        current = ""
        for raw in parts:
            part = raw.strip()
            if not part:
                continue
            candidate = part if not current else f"{current} {part}"
            if current and len(self.encode(candidate)) > limit:
                result.append(current)
                current = part
            else:
                current = candidate
        if current:
            result.append(current)
        return result

    def _split_words(self, text: str, limit: int) -> list[str]:
        result: list[str] = []
        current: list[str] = []
        for word in text.split():
            candidate = " ".join([*current, word])
            if current and len(self.encode(candidate)) > limit:
                result.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
            if len(self.encode(" ".join(current))) > limit:
                raise ValueError(
                    "A single Pocket tokenization unit exceeds max_token_per_chunk; "
                    "cannot split safely by whitespace"
                )
        if current:
            result.append(" ".join(current))
        return result

import json

import pytest

from pocketsynth.bundle import BundleMetadata
from pocketsynth.frontend import PocketFrontend


class FakeProcessor:
    def EncodeAsIds(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class CharacterProcessor:
    def EncodeAsIds(self, text: str) -> list[int]:
        return list(range(len(text)))


def make_metadata(tmp_path, **options) -> BundleMetadata:
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "bundle_name": "test",
                "language": "en",
                "schema_version": 2,
                "sample_rate": 24_000,
                "samples_per_frame": 1920,
                "max_token_per_chunk": options.pop("max_token_per_chunk", 4),
                "tokenizer_file": "tokenizer.model",
                "bos_before_voice_file": "bos.npy",
                **options,
            }
        )
    )
    return BundleMetadata.load(path)


def test_frontend_splits_only_when_token_limit_requires_it(tmp_path) -> None:
    metadata = make_metadata(tmp_path)
    frontend = PocketFrontend("unused", metadata, processor=FakeProcessor())

    assert frontend.split_for_model("one two three") == ("one two three",)
    chunks = frontend.split_for_model("one two three four five six")
    assert chunks == ("one two three four", "five six")
    assert all(len(frontend.encode(chunk)) <= 4 for chunk in chunks)


def test_frontend_applies_only_pocket_model_normalization(tmp_path) -> None:
    metadata = make_metadata(
        tmp_path,
        remove_semicolons=True,
        pad_with_spaces_for_short_inputs=True,
    )
    frontend = PocketFrontend("unused", metadata, processor=FakeProcessor())

    assert frontend.prepare_text("  Pay $12.50;  then wait.  ") == " Pay $12.50, then wait. "
    assert frontend.encode("Pay $12.50") == (0, 1)


def test_frontend_rejects_single_oversized_token_unit(tmp_path) -> None:
    frontend = PocketFrontend(
        "unused",
        make_metadata(tmp_path, max_token_per_chunk=4),
        processor=CharacterProcessor(),
    )

    with pytest.raises(ValueError, match="single Pocket tokenization unit"):
        frontend.split_for_model("oversized")

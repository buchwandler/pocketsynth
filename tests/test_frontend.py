import json

from pocketsynth.bundle import BundleMetadata
from pocketsynth.frontend import PocketFrontend


class FakeProcessor:
    def EncodeAsIds(self, text):
        return list(range(len(text.split())))


def test_frontend_splits_only_when_token_limit_requires_it(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({
        "bundle_name": "test",
        "language": "en",
        "schema_version": 2,
        "sample_rate": 24000,
        "samples_per_frame": 1920,
        "max_token_per_chunk": 4,
        "tokenizer_file": "tokenizer.model",
        "bos_before_voice_file": "bos.npy",
    }))
    metadata = BundleMetadata.load(path)
    frontend = PocketFrontend("unused", metadata, processor=FakeProcessor())
    assert frontend.split_for_model("one two three") == ("one two three",)
    chunks = frontend.split_for_model("one two three four five six")
    assert chunks == ("one two three four", "five six")
    assert all(len(frontend.encode(chunk)) <= 4 for chunk in chunks)

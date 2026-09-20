import json
from unittest.mock import MagicMock

import numpy as np

from pocketsynth.audio_job import render_span
from pocketsynth.bundle import BundleMetadata
from pocketsynth.config import GenerationConfig
from pocketsynth.frontend import PocketFrontend
from pocketsynth.plan_adapter import PreparedPocketSpan
from tests.fakes import make_test_voice


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

def test_long_input_is_split_before_runtime_invocation(tmp_path):
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
    runtime = type("Runtime", (), {})()
    runtime.frontend = frontend
    runtime.sample_rate = 24000
    runtime.infer_tokens = MagicMock(return_value=np.ones(3, dtype=np.float32))
    span = PreparedPocketSpan(
        id="span",
        text="one two three four five six",
        language="en",
        segment_ids=(),
        spoken_start=0,
        spoken_end=27,
        pause_before_seconds=0.0,
        pause_after_seconds=0.0,
        voice_ref=None,
    )

    render_span(
        span,
        runtime=runtime,
        voice=make_test_voice(),
        generation=GenerationConfig(),
    )

    assert runtime.infer_tokens.call_count == 2
    assert all(len(call.args[0]) <= 4 for call in runtime.infer_tokens.call_args_list)

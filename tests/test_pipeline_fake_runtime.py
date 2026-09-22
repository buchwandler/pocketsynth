from __future__ import annotations

import json

from pocketsynth.config import PipelineConfig
from pocketsynth.pipeline import PocketPipeline
from tests.fakes import FakeBundleMetadata, FakePocketRuntime


def test_pipeline_fake_runtime_prepares_a_voice_once(tmp_path) -> None:
    (tmp_path / "bundle.json").write_text(
        json.dumps(
            {
                "bundle_name": "fake-pocket",
                "language": "en",
                "schema_version": 2,
                "sample_rate": 24_000,
                "samples_per_frame": 1_920,
                "max_token_per_chunk": 50,
                "tokenizer_file": "tokenizer.model",
                "bos_before_voice_file": "bos.npy",
            }
        )
    )
    runtime = FakePocketRuntime(metadata=FakeBundleMetadata())
    pipeline = PocketPipeline(PipelineConfig(bundle_dir=tmp_path), runtime=runtime)

    with pipeline:
        prepared = pipeline.set_default_voice("reference.wav")
        assert pipeline.prepare_voice(prepared) is prepared

    assert runtime._prepare_count == 1
    assert runtime._closed

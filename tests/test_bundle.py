import json
from pathlib import Path

from pocketsynth.bundle import BundleMetadata, BundlePaths


def _bundle(tmp_path: Path) -> Path:
    data = {
        "bundle_name": "english_2026-04",
        "language": "english_2026-04",
        "schema_version": 2,
        "sample_rate": 24000,
        "samples_per_frame": 1920,
        "max_token_per_chunk": 50,
        "tokenizer_file": "tokenizer.model",
        "bos_before_voice_file": "bos_before_voice.npy",
        "predefined_voices": ["alba"],
    }
    (tmp_path / "bundle.json").write_text(json.dumps(data))
    for name in (
        "tokenizer.model",
        "bos_before_voice.npy",
        "flow_lm_main_int8.onnx",
        "flow_lm_flow_int8.onnx",
        "mimi_decoder_int8.onnx",
        "flow_lm_main.onnx",
        "flow_lm_flow.onnx",
        "mimi_decoder.onnx",
        "mimi_encoder.onnx",
        "text_conditioner.onnx",
    ):
        (tmp_path / name).write_bytes(b"x")
    return tmp_path


def test_metadata_and_int8_profile(tmp_path):
    root = _bundle(tmp_path)
    metadata = BundleMetadata.load(root / "bundle.json")
    assert metadata.max_token_per_chunk == 50
    paths = BundlePaths.from_directory(root, precision="int8")
    assert paths.flow_lm_main.name.endswith("_int8.onnx")
    assert paths.mimi_encoder.name == "mimi_encoder.onnx"
    assert paths.text_conditioner.name == "text_conditioner.onnx"

[![PyPI - Version](https://img.shields.io/pypi/v/pocketsynth)](https://pypi.org/project/pocketsynth/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pocketsynth)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pocketsynth)
[![codecov](https://codecov.io/gh/buchwandler/pocketsynth/graph/badge.svg?token=ticsUMNuF8)](https://codecov.io/gh/buchwandler/pocketsynth)

# pocketsynth

`pocketsynth` is an UtterPlan-aware Python runtime for Pocket TTS ONNX bundles. It turns text and a reference voice WAV into a WAV file while delegating asset management and model execution to OnnxVoice.

## Status and installation

PocketSynth requires the Pocket-capable OnnxVoice release:

```bash
python -m pip install -e '.[cpu]'
```

The package declares `onnxvoice>=0.1.10,<0.2` and `utterplan>=0.2.0,<0.3`. OnnxVoice owns the Pocket catalog, downloads, cache, ONNX Runtime sessions, Mimi, Flow-LM, and decoding. PocketSynth does not duplicate those responsibilities.

## Quickstart

Provide a mono, 16-bit PCM reference voice WAV. The managed bundle is downloaded and cached by OnnxVoice:

```bash
python examples/quickstart.py --voice reference.wav --output hello.wav
```

The equivalent Python API is:

```python
from pocketsynth import synthesize_to_wav

synthesize_to_wav(
    "Hello from Pocket.",
    "hello.wav",
    bundle="english_2026-04",
    voice="reference.wav",
)
```

The CLI provides the same path:

```bash
pocketsynth synthesize \
  --bundle english_2026-04 \
  --voice reference.wav \
  --output hello-cli.wav \
  "Hello from Pocket."
```

Use `--cache-dir` to select the managed cache. A cached second run can use `--offline`. `--refresh-catalog` and `--force-download` make asset selection reproducible when refreshing or replacing cached assets.

## Local bundle

A local bundle avoids catalog resolution and network access. Use the semantic bundle directory supplied by OnnxVoice:

```bash
python examples/local_bundle.py \
  --bundle-dir ./onnx/english_2026-04 \
  --voice reference.wav \
  --output hello-local.wav
```

The Python form is:

```python
from pocketsynth import PocketPipeline

with PocketPipeline.load("./onnx/english_2026-04") as tts:
    tts.set_default_voice("reference.wav")
    tts("Hello from Pocket.").save_wav("hello-local.wav")
```

PocketSynth locates the high-level bundle metadata and component files, then asks OnnxVoice to open the runtime. It does not open ONNX sessions directly.

## Reference voice policy

The application boundary accepts a mono PCM WAV with 16-bit samples. Invalid prompts report the actual channels, sample width, sample rate, and compression. Valid prompts at another sample rate are resampled to the bundle rate.

## Reusing a voice

```python
from pocketsynth import PocketPipeline

with PocketPipeline.from_pretrained("english_2026-04") as tts:
    narrator = tts.prepare_voice("narrator.wav")
    tts.set_default_voice(narrator)
    first = tts("First sentence.")
    second = tts("Second sentence.")
```

The voice encoder runs once during `prepare_voice()`. `PreparedVoice` hides the OnnxVoice state representation and can be reused for later calls on the same bundle.

## Ownership

UtterPlan owns semantic units. PocketSynth owns text preparation, SentencePiece tokenization, token-limit subdivision, voice bindings, generation settings, composition, and WAV output. OnnxVoice owns catalog resolution, downloads, cache integrity, provider sessions, bundle graph contracts, voice encoding, inference, and decoding.

```text
UtterPlan -> PocketSynth policy -> OnnxVoice Pocket runtime -> AudioCompose -> WAV
```

## Examples

See [`examples/README.md`](examples/README.md). The runner avoids managed downloads unless requested:

```bash
python examples/run_all.py --list
python examples/run_all.py --local-only
python examples/run_all.py --include-network
python examples/run_all.py --include-network --offline
```

Set `POCKETSYNTH_EXAMPLE_VOICE` and, for local examples, `POCKETSYNTH_EXAMPLE_BUNDLE_DIR` as described in the examples documentation.

## Diagnostics and tests

The check command reports actionable dependency, provider, catalog or local bundle, and reference voice checks:

```bash
pocketsynth check --provider CPUExecutionProvider
pocketsynth check --bundle-dir ./onnx/english_2026-04 --voice reference.wav
```

Run the package checks with:

```bash
pytest -q
ruff check pocketsynth tests examples
python -m compileall -q pocketsynth examples
```

Real model checks are environment-gated:

```bash
export POCKETSYNTH_TEST_BUNDLE_DIR=/path/to/onnx/english_2026-04
export POCKETSYNTH_TEST_VOICE_WAV=/path/to/reference.wav
pytest -q -m integration tests/integration/test_real_local_wav.py
```

The managed integration test additionally requires network access and uses `POCKETSYNTH_TEST_BUNDLE` when set. It verifies a cached rerun with explicit offline mode.

## Current limitations

- Named predefined voice states are not downloaded by PocketSynth.
- Streaming remains at UtterPlan sentence-unit granularity.
- Real integration tests require external bundles and a reference recording.

[![PyPI - Version](https://img.shields.io/pypi/v/pocketsynth)](https://pypi.org/project/pocketsynth/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pocketsynth)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pocketsynth)
[![codecov](https://codecov.io/gh/buchwandler/pocketsynth/graph/badge.svg?token=ticsUMNuF8)](https://codecov.io/gh/buchwandler/pocketsynth)

# pocketsynth

`pocketsynth` is an UtterPlan-aware Python runtime for Pocket TTS ONNX bundles. It turns text and a bundle-declared predefined voice or reference WAV into a WAV file while delegating asset management and model execution to OnnxVoice.

## Status and installation

PocketSynth requires the Pocket-capable OnnxVoice release:

```bash
python -m pip install -e '.[cpu]'
```

The package declares `onnxvoice>=0.1.11,<0.2` and `utterplan>=0.2.0,<0.3`. OnnxVoice owns the Pocket catalog, downloads, cache, ONNX Runtime sessions, Mimi, Flow-LM, and decoding. PocketSynth does not duplicate those responsibilities.

## Quickstart

The first smoke test uses the bundle-declared predefined voice `alba`, without a reference WAV:

```bash
python examples/predefined_voice.py
```

The managed CLI path is:

```bash
pocketsynth synthesize --bundle english_2026-04 --voice alba --output hello.wav "Hello from Pocket."
```

The equivalent Python API is:

```python
from pocketsynth import synthesize_to_wav

synthesize_to_wav(
    "Hello from Pocket.",
    "hello.wav",
    bundle="english_2026-04",
    voice="alba",
)
```

`PocketPipeline.predefined_voices` lists the names declared by the selected bundle. A declaration means the voice is compatible with that bundle, not that its separate upstream state asset is available or that your account has access to it. Some voice-state assets are gated. Accept the upstream access terms and configure Hugging Face authentication before the first online run. OnnxVoice owns voice downloads and caching.

The `pocketsynth[cpu]` and `pocketsynth[gpu]` extras include OnnxVoice's optional Hugging Face and Safetensors support. For a gated voice, accept access on the upstream model page and authenticate the same account used by the application with `hf auth login` or `HF_TOKEN`. OnnxVoice does not initiate login or store tokens. To inspect local package, credential, and offline status without a network request, run `onnxvoice doctor --system pocket`; it does not test gated-repository access. See the OnnxVoice Pocket download guide for more details.

`pocketsynth check --bundle english_2026-04 --voice alba` validates the bundle's declared name without installing model files.

Both the bundle files and voice state must be cached before using `--offline`:

```bash
pocketsynth synthesize --bundle english_2026-04 --voice alba --offline --output hello-offline.wav "Hello from Pocket."
```

Use `--cache-dir` to select the managed cache. `--refresh-catalog` and `--force-download` control bundle asset refresh and replacement.

## Reference-WAV cloning

Reference-WAV cloning remains supported as a second voice source. Provide a mono, 16-bit PCM WAV:

```bash
python examples/quickstart.py --voice reference.wav --output hello.wav
```

The CLI and Python API also accept the WAV path directly:

```bash
pocketsynth synthesize \
  --bundle english_2026-04 \
  --voice reference.wav \
  --output hello-cli.wav \
  "Hello from Pocket."
```

```python
from pocketsynth import synthesize_to_wav

synthesize_to_wav(
    "Hello from Pocket.",
    "hello.wav",
    bundle="english_2026-04",
    voice="reference.wav",
)
```

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

## Voice source policy

A bundle's `predefined_voice_names` declares compatibility with its model only. It does not guarantee that the separate upstream voice-state asset is available or that your account can access it. OnnxVoice owns state resolution, downloads, authentication, and caching. Reference prompts remain mono PCM WAVs with 16-bit samples. Invalid prompts report the actual channels, sample width, sample rate, and compression. Valid prompts at another sample rate are resampled to the bundle rate.

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

Set `POCKETSYNTH_EXAMPLE_VOICE` for reference-WAV examples and, for local examples, `POCKETSYNTH_EXAMPLE_BUNDLE_DIR` as described in the examples documentation.

## Diagnostics and tests

The check command reports dependencies, providers, bundle metadata, and predefined names or reference WAVs:

```bash
pocketsynth check --provider CPUExecutionProvider
pocketsynth check --bundle english_2026-04 --voice alba
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

The managed reference-WAV integration test uses `POCKETSYNTH_TEST_BUNDLE` when set. It verifies a cached rerun with explicit offline mode.

The predefined `alba` integration test is explicitly gated. Enable it only after gated Hugging Face access and credentials are configured. It verifies online and cached-offline synthesis with the same cache, including mono, 16-bit PCM, 24 kHz, non-silent WAV output. Without the opt-in variable, the test skips.

```bash
export POCKETSYNTH_TEST_PREDEFINED_VOICE=1
pytest -q -m integration tests/integration/test_real_predefined_voice_wav.py
```

## Current limitations

- Predefined voice-state assets are separate from bundle files, may be gated, and are managed by OnnxVoice.
- Streaming remains at UtterPlan sentence-unit granularity.
- Real integration tests require external bundles; reference-WAV tests additionally require a reference recording.

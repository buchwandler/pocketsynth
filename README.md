# pocketsynth

`pocketsynth` is an UtterPlan-aware Python runtime for Pocket TTS ONNX bundles. It provides the application-facing path from text and a reference voice to a WAV file while delegating model assets and graph execution to OnnxVoice.

## Status

Pocket support is implemented across PocketSynth and OnnxVoice. Release readiness depends on the OnnxVoice Pocket catalog contract and real v2 runtime parity tests.

The minimum compatible dependency is `onnxvoice>=0.2,<0.3`. PocketSynth does not duplicate OnnxVoice catalog, download, cache, ORT session, Mimi, Flow-LM, or decoder logic.

## Installation

```bash
pip install "pocketsynth[cpu]"
```

## First WAV, managed bundle

Provide a mono 16-bit PCM reference WAV:

```python
from pocketsynth import synthesize_to_wav

path = synthesize_to_wav(
    "Hello from PocketSynth.",
    "hello.wav",
    bundle="english_2026-04",
    voice="reference.wav",
)
print(path)
```

The same flow is available from the CLI:

```bash
pocketsynth synthesize \
  --bundle english_2026-04 \
  --voice reference.wav \
  --output hello-cli.wav \
  "Hello world."
```

Managed assets are cached by OnnxVoice. Use `--cache-dir` to select a cache and `--offline` for an explicitly cached second run.

## First WAV, local bundle

```python
from pocketsynth import PocketPipeline

with PocketPipeline.load("./onnx/english_2026-04", precision="int8") as tts:
    tts.set_default_voice("reference.wav")
    tts("Hello from Pocket.").save_wav("hello-local.wav")
```

```bash
pocketsynth synthesize \
  --bundle-dir ./onnx/english_2026-04 \
  --voice reference.wav \
  --output hello-local-cli.wav \
  "Hello world."
```

Local opening is network-free and requires the concrete bundle files expected by OnnxVoice.

## Reference WAV policy

The application boundary accepts mono PCM WAV with 16-bit samples. Invalid prompts report the actual channel count, sample width, sample rate, and compression. Other sample rates are resampled to the bundle rate by PocketSynth.

## Reusing a voice

```python
with PocketPipeline.from_pretrained("english_2026-04") as tts:
    narrator = tts.prepare_voice("narrator.wav")
    tts.set_default_voice(narrator)
    first = tts("First sentence.")
    second = tts("Second sentence.")
```

The Mimi encoder runs once for `prepare_voice()`. The resulting `PreparedVoice` is reusable without exposing the internal OnnxVoice state shape.

## Planning and ownership

UtterPlan owns semantic units. PocketSynth applies Pocket text normalization, SentencePiece tokenization, token-limit subdivision, voice bindings, generation settings, and composition. OnnxVoice owns catalog resolution, downloads, cache integrity, provider sessions, bundle graph contracts, voice encoding, inference, and decoding.

```text
UtterPlan -> PocketSynth frontend/policy -> OnnxVoice Pocket runtime -> AudioCompose -> WAV
```

Use `tts.plan()` and `tts.render_plan()` for explicit plan-first workflows. `examples/basic.py` remains the plan-first managed example.

## Examples

Start with the success path in [`examples/README.md`](examples/README.md):

```bash
POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference.wav python examples/first_wav.py
POCKETSYNTH_EXAMPLE_BUNDLE_DIR=/path/to/bundle \
POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference.wav \
python examples/first_wav_local.py
```

The runner avoids managed downloads by default:

```bash
python examples/run_all.py --list
python examples/run_all.py --local-only
python examples/run_all.py --include-network
python examples/run_all.py --include-network --offline
```

Validate a generated container without an additional audio library:

```bash
python - <<'PY'
import wave
with wave.open("example-artefacts/first_wav.wav", "rb") as f:
    print(f.getframerate(), f.getnframes())
PY
```

## Diagnostics

```bash
pocketsynth check --provider CPUExecutionProvider
pocketsynth check --bundle-dir ./onnx/english_2026-04 --voice reference.wav
```

The check command reports OnnxVoice import/version, available ORT providers, local bundle artifacts or managed catalog resolution, and reference WAV validity.

## Tests

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

The managed test additionally requires network access and uses `POCKETSYNTH_TEST_BUNDLE` when set.

## Current limitations

- Named predefined voice states are not downloaded by PocketSynth.
- Streaming remains at UtterPlan sentence-unit granularity.
- Real integration tests require external bundles and a reference recording.

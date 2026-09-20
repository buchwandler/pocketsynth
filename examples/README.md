# PocketSynth examples

These examples are ordered from the smallest successful first-WAV path to the more explicit planning APIs.

## Prerequisites

```bash
pip install -e ".[cpu]"
```

Set a mono 16-bit PCM reference WAV for examples that synthesize audio:

```bash
export POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference.wav
```

Managed examples use the OnnxVoice catalog and default to `english_2026-04`. Local examples additionally need:

```bash
export POCKETSYNTH_EXAMPLE_BUNDLE_DIR=/path/to/onnx/english_2026-04
```

## 1. First WAV, managed

The smallest managed path is `first_wav.py`:

```bash
python examples/first_wav.py
```

It writes `example-artefacts/first_wav.wav`.

For visible download progress, use:

```bash
python examples/download_and_synthesize.py
```

## 2. First WAV, local

`first_wav_local.py` skips catalog resolution and plan persistence:

```bash
python examples/first_wav_local.py
```

A local failure points at the bundle or runtime. If local succeeds but managed fails, inspect catalog and installation behavior.

## 3. Explicit UtterPlan example

`basic.py` demonstrates planning, saving, and rendering a managed bundle. `basic_local.py` is the corresponding network-free path.

## 4. Prepared voice reuse

`pretrained_pipeline.py` prepares one voice and renders two sentences without re-encoding the voice:

```bash
python examples/pretrained_pipeline.py
```

## 5. Plan round-trip

`plan_roundtrip.py` saves an UtterancePlan, reloads it, and renders the reloaded plan.

## 6. Run all examples

The runner avoids surprise multi-hundred-megabyte downloads by default:

```bash
python examples/run_all.py --list
python examples/run_all.py --local-only
python examples/run_all.py --include-network
python examples/run_all.py --managed-only --include-network
python examples/run_all.py --include-network --offline
python examples/run_all.py --include-network --fail-fast
```

Managed execution requires `--include-network`, except when using `--offline` with assets already cached by OnnxVoice. Local execution requires `POCKETSYNTH_EXAMPLE_BUNDLE_DIR`.

The runner validates every generated WAV as mono 16-bit PCM with a positive frame count and non-silent samples. It validates UtterPlan files with UtterPlan itself.

## WAV container verification

No additional audio library is needed to inspect the container:

```bash
python - <<'PY'
import wave
with wave.open("example-artefacts/first_wav.wav", "rb") as f:
    print(f.getframerate(), f.getnframes())
PY
```

All scripts honor `POCKETSYNTH_EXAMPLE_OUTPUT_DIR`. Managed scripts honor `POCKETSYNTH_EXAMPLE_OFFLINE=1`.

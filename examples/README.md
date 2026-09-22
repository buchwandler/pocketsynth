# PocketSynth examples

These examples start with the shortest managed and local paths, then show planning and progress APIs.

## Prerequisites

Install the Pocket-capable runtime:

```bash
python -m pip install -e '.[cpu]'
```

Every synthesis example needs a mono, 16-bit PCM reference WAV:

```bash
export POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference.wav
```

Managed examples use the `english_2026-04` catalog bundle by default. Local examples also need:

```bash
export POCKETSYNTH_EXAMPLE_BUNDLE_DIR=/path/to/onnx/english_2026-04
```

## 1. Quickstart

The canonical first smoke uses the managed catalog and does not require manual model-file downloads:

```bash
python examples/quickstart.py \
  --voice /path/to/reference.wav \
  --output hello.wav
```

Use `--cache-dir` to select the OnnxVoice cache. A cached rerun can add `--offline`.

The local equivalent opens a bundle directory through OnnxVoice:

```bash
python examples/local_bundle.py \
  --bundle-dir /path/to/onnx/english_2026-04 \
  --voice /path/to/reference.wav \
  --output hello-local.wav
```

`quickstart.py` and `local_bundle.py` select semantic bundle references and directories only. They do not select ONNX filenames.

## 2. Basic plan-first flow

`basic.py` demonstrates planning, saving, and rendering a managed bundle. `basic_local.py` is the corresponding network-free path.

```bash
python examples/basic.py
python examples/basic_local.py
```

## 3. Managed download progress

`download_and_synthesize.py` uses the stable PocketSynth progress callback and console renderer:

```bash
python examples/download_and_synthesize.py
```

## 4. Prepared voice reuse

`pretrained_pipeline.py` prepares one voice and renders multiple sentences without re-encoding it:

```bash
python examples/pretrained_pipeline.py
```

## 5. Plan round-trip

`plan_roundtrip.py` saves an UtterancePlan, reloads it, and renders the reloaded plan:

```bash
python examples/plan_roundtrip.py
```

## 6. Run all examples

The runner avoids surprise managed downloads by default:

```bash
python examples/run_all.py --list
python examples/run_all.py --local-only
python examples/run_all.py --include-network
python examples/run_all.py --managed-only --include-network
python examples/run_all.py --include-network --offline
python examples/run_all.py --include-network --fail-fast
```

Managed execution requires `--include-network`, except for `--offline` runs with assets already cached by OnnxVoice. Local execution requires `POCKETSYNTH_EXAMPLE_BUNDLE_DIR`. The runner validates generated WAV containers as mono, 16-bit PCM with positive, non-silent audio, and validates saved UtterPlan files.

## WAV verification

No additional audio library is needed to inspect a generated container:

```bash
python - <<'PY'
import wave

with wave.open("hello.wav", "rb") as stream:
    print(stream.getframerate(), stream.getnframes())
PY
```

All scripts honor `POCKETSYNTH_EXAMPLE_OUTPUT_DIR`. Managed scripts honor `POCKETSYNTH_EXAMPLE_OFFLINE=1` where applicable.

# PocketSynth Examples

This directory contains runnable examples demonstrating PocketSynth usage.

## Prerequisites

Install PocketSynth and its dependencies:

```bash
pip install -e ".[cpu]"
```

For local examples, download a Pocket ONNX bundle and prepare a reference voice WAV.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `POCKETSYNTH_EXAMPLE_VOICE` | Yes | Path to a mono 16-bit PCM reference WAV |
| `POCKETSYNTH_EXAMPLE_BUNDLE_DIR` | For local | Path to a local Pocket ONNX bundle directory |
| `POCKETSYNTH_EXAMPLE_BUNDLE` | For managed | Managed bundle name (default: `english_2026-04`) |
| `POCKETSYNTH_EXAMPLE_OUTPUT_DIR` | No | Custom output directory (default: `example-artefacts/`) |

## Examples

### basic_local.py

Smallest local bundle example. Requires `POCKETSYNTH_EXAMPLE_BUNDLE_DIR`.

```bash
export POCKETSYNTH_EXAMPLE_BUNDLE_DIR=/path/to/pocket-tts-onnx/onnx/english_2026-04
export POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference_sample.wav
python examples/basic_local.py
```

### basic.py

Managed bundle example. Requires OnnxVoice Pocket catalog support.

```bash
export POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference_sample.wav
python examples/basic.py
```

### pretrained_pipeline.py

Demonstrates prepared-voice reuse: prepare once, synthesize multiple sentences.

### plan_roundtrip.py

Demonstrates semantic reproducibility: plan, save, reload, render.

### run_all.py

Run all examples and validate outputs:

```bash
python examples/run_all.py
```

Options:
- `--list`: List available examples
- `--fail-fast`: Stop on first failure

## Output

Examples write artifacts to `example-artefacts/` (or `POCKETSYNTH_EXAMPLE_OUTPUT_DIR`).

Each example produces:
- A `.utterplan.json` file (the semantic plan)
- A `.wav` file (the synthesized audio)

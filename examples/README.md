# PocketSynth examples

These examples demonstrate PocketSynth as a synthesis engine. They pass already-prepared speakable text, a Pocket bundle, and a concrete voice source to `PocketRuntime` or the convenience API. They do not parse documents or create plans.

## Prerequisites

Install the Pocket-capable runtime:

```bash
python -m pip install -e '.[cpu]'
```

`predefined_voice.py` uses the bundle-declared voice `alba` and does not need a reference WAV. Other examples that use a prompt voice require a mono, 16-bit PCM reference WAV:

```bash
export POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference.wav
```

Managed examples use `english_2026-04` by default. Local examples also need a bundle directory:

```bash
export POCKETSYNTH_EXAMPLE_BUNDLE_DIR=/path/to/onnx/english_2026-04
```

## Managed predefined voice

```bash
python examples/predefined_voice.py
```

The script writes `example-artefacts/predefined_voice_alba.wav`. The bundle's `predefined_voice_names` declares compatibility only. Voice-state assets are separate, may require accepted Hugging Face access terms and authentication, and are resolved and cached by OnnxVoice. After the bundle and voice state are cached, rerun with `POCKETSYNTH_EXAMPLE_OFFLINE=1` to avoid network access.

## Long-text synthesis

`long_text.py` uses the bundle-declared `alba` voice and demonstrates both the default Phrasplit sentence boundaries and `sentence_split="none"`. It writes `long-text.wav` and `long-text-no-split.wav`. The `none` mode is useful when an application already owns segmentation or wants to minimize preprocessing.

```bash
python examples/long_text.py
```

Managed assets are fetched by default. Set `POCKETSYNTH_EXAMPLE_OFFLINE=1` to use cached assets only.

## Reference-WAV synthesis

The managed CLI-style example uses a reference WAV:

```bash
python examples/quickstart.py \
  --voice /path/to/reference.wav \
  --output hello.wav
```

The local equivalent opens a concrete bundle directory:

```bash
python examples/local_bundle.py \
  --bundle-dir /path/to/onnx/english_2026-04 \
  --voice /path/to/reference.wav \
  --output hello-local.wav
```

## Reusing a prepared voice

`basic.py` and `basic_local.py` prepare a voice once and synthesize one independent request. Applications can retain a `PreparedVoice` and reuse it for many requests on the same compatible runtime.

```bash
python examples/basic.py
python examples/basic_local.py
```

## Managed download progress

`download_and_synthesize.py` uses the stable PocketSynth progress callback and console renderer:

```bash
python examples/download_and_synthesize.py
```

## Run examples

The runner avoids managed downloads by default. If `POCKETSYNTH_EXAMPLE_VOICE` is unset, it can run the predefined-voice example and skips examples that need a reference WAV. Listing examples does not run them.

```bash
python examples/run_all.py --list
python examples/run_all.py --local-only
python examples/run_all.py --include-network
python examples/run_all.py --managed-only --include-network
python examples/run_all.py --include-network --offline
python examples/run_all.py --include-network --fail-fast
```

Managed execution requires `--include-network`; add `--offline` to restrict it to assets already cached by OnnxVoice. Local execution requires `POCKETSYNTH_EXAMPLE_BUNDLE_DIR`. The runner verifies generated WAV files are mono, 16-bit PCM, and contain non-silent audio.

## Inspect a WAV

No additional audio library is needed to inspect a generated container:

```bash
python - <<'PY'
import wave

with wave.open("hello.wav", "rb") as stream:
    print(stream.getframerate(), stream.getnframes())
PY
```

All scripts honor `POCKETSYNTH_EXAMPLE_OUTPUT_DIR`. Managed scripts honor `POCKETSYNTH_EXAMPLE_OFFLINE=1` where applicable.

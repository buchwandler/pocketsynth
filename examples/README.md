# PocketSynth examples

These examples demonstrate PocketSynth as a synthesis engine. They pass already-prepared speakable text, a Pocket bundle, and a concrete voice source to `PocketRuntime` or the convenience API. They do not parse documents or create plans.

## Prerequisites

Install the Pocket-capable runtime:

```bash
python -m pip install -e '.[cpu]'
```

`predefined_voice.py` uses the bundle-declared voice `alba` and does not need a reference WAV. `kyutai_voice.py` uses a managed catalog prompt and also needs no local WAV. Examples that use a local prompt accept uncompressed PCM WAV at 8, 16, 24, or 32 bits, with mono or multichannel input (multichannel audio is downmixed):

```bash
export POCKETSYNTH_EXAMPLE_VOICE=/path/to/reference.wav
```

Managed examples use `english_2026-04` by default. Local examples also need a bundle directory:

```bash
export POCKETSYNTH_EXAMPLE_BUNDLE_DIR=/path/to/onnx/english_2026-04
```

Unless an explicit output path or `POCKETSYNTH_EXAMPLE_OUTPUT_DIR` is supplied, every example writes WAV files under `example-artifacts/`.

## Managed predefined voice

```bash
python examples/predefined_voice.py
```

The script writes `example-artifacts/predefined_voice_alba.wav`. The bundle's `predefined_voice_names` declares compatibility only. Voice-state assets are separate, may require accepted Hugging Face access terms and authentication, and are resolved and cached by OnnxVoice. After the bundle and voice state are cached, rerun with `POCKETSYNTH_EXAMPLE_OFFLINE=1` to avoid network access.

## Managed Kyutai reference voices

`kyutai_voice.py` demonstrates the simple current-reference workflow without a manual WAV download. Add `--pin-prompt` to inspect and retain catalog metadata first, then prepare exactly that identity:

```bash
python examples/kyutai_voice.py
python examples/kyutai_voice.py --pin-prompt \
  --output example-artifacts/kyutai-casual-pinned.wav
```

The default prepares the current catalog identity for `kyutai-tts-voices:alba-mackenna/casual`. `--pin-prompt` calls `inspect_voice_prompt()` before opening the runtime, then passes the resulting `VoicePromptInfo` to `prepare_voice()`. Inspection reads metadata only. Pinned preparation verifies the source SHA-256 and prompt revision before OnnxVoice fetches the WAV; a changed prompt raises `VoicePromptChangedError`. `PreparedVoice.voice_prompt` retains prompt provenance, while `bundle_revision` remains the model revision. The catalog WAV SHA and normalized prepared-audio fingerprint are distinct. After the bundle and prompt are cached, run either mode with `--offline` to reuse them without network access. The CLI accepts the same managed reference:

```bash
pocketsynth synthesize \
  --bundle english_2026-04 \
  --voice kyutai-tts-voices:alba-mackenna/casual \
  --output example-artifacts/kyutai-casual.wav \
  "Hello from PocketSynth."

pocketsynth synthesize --offline \
  --bundle english_2026-04 \
  --voice kyutai-tts-voices:alba-mackenna/casual \
  --output example-artifacts/kyutai-casual-offline.wav \
  "This prompt is loaded from cache."
```

List catalog metadata without downloading prompt WAVs:

```bash
pocketsynth voices list --dataset alba-mackenna
```

`clone_all_kyutai_voices.py` discovers every cataloged prompt by default, continues after per-prompt failures, and writes `manifest.csv` in the output directory:

```bash
python examples/clone_all_kyutai_voices.py \
  --bundle english_2026-04 \
  --output-dir example-artifacts/kyutai-voices
```

Use `--dataset`, `--variant`, and `--limit` to select a smaller set. `--dry-run` writes the planned outputs to the manifest without opening a synthesis runtime or fetching WAVs. `--offline` uses cached catalog, bundle, and prompt assets. `--exclude-noncommercial` mechanically filters explicit license labels from the catalog; this is not legal advice.

## Long-text synthesis

`long_text.py` uses the bundle-declared `alba` voice and demonstrates opt-in Phrasplit sentence boundaries and `sentence_split="none"`. Sentence splitting defaults to `none`, which still applies Pocket model-limit chunking. It writes `example-artifacts/long-text.wav` and `example-artifacts/long-text-no-split.wav`.

```bash
python examples/long_text.py
```

Managed assets are fetched by default. Set `POCKETSYNTH_EXAMPLE_OFFLINE=1` to use cached assets only.

## Token length and semantic-shape matrix

`token_shape_matrix.py` creates a controlled set of Pocket renders that varies semantic shape (one sentence vs multiple sentences) and real model-token count. It measures each candidate with the active Pocket tokenizer rather than estimating from characters or words.

Default token targets are 10, 20, 30, 40, 45, 50, and the active bundle maximum plus 5. For the current English bundle, the final `>50` target resolves to 55 tokens; other bundles resolve it dynamically.

For each text, the diagnostic renders:

- `sentence_split="none"`
- `sentence_split="phrasplit"`

The output directory contains paired WAVs, `manifest.csv`, `manifest.json`, `texts.json`, and `summary.md`. This model-backed diagnostic is intentionally not part of the lightweight default example runner.

```bash
python examples/token_shape_matrix.py
```

Use `--measure-only` to generate measured text and manifests without rendering, `--require-exact-targets` to fail when a requested bucket cannot be matched, or `--output-dir PATH` to choose an output location. Managed assets are fetched by default; set `POCKETSYNTH_EXAMPLE_OFFLINE=1` to use cached assets only.

## Reference-WAV synthesis

The managed CLI-style example uses a reference WAV:

```bash
python examples/quickstart.py \
  --voice /path/to/reference.wav \
  --output example-artifacts/quickstart.wav
```

The local equivalent opens a concrete bundle directory:

```bash
python examples/local_bundle.py \
  --bundle-dir /path/to/onnx/english_2026-04 \
  --voice /path/to/reference.wav \
  --output example-artifacts/local_bundle.wav
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

The runner avoids managed downloads by default. With `--include-network` and no `POCKETSYNTH_EXAMPLE_VOICE`, it can run the predefined-voice and single managed Kyutai examples while skipping scripts that need a local WAV. Listing examples does not run them.

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

with wave.open("example-artifacts/quickstart.wav", "rb") as stream:
    print(stream.getframerate(), stream.getnframes())
PY
```

All scripts honor `POCKETSYNTH_EXAMPLE_OUTPUT_DIR`. Managed scripts honor `POCKETSYNTH_EXAMPLE_OFFLINE=1` where applicable.

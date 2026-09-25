[![PyPI - Version](https://img.shields.io/pypi/v/pocketsynth)](https://pypi.org/project/pocketsynth/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pocketsynth)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pocketsynth)
[![codecov](https://codecov.io/gh/buchwandler/pocketsynth/graph/badge.svg?token=ticsUMNuF8)](https://codecov.io/gh/buchwandler/pocketsynth)

# PocketSynth

PocketSynth is a Python synthesis engine for Pocket TTS ONNX bundles. It owns Pocket-specific text normalization, SentencePiece tokenization, model-limit chunking, voice-prompt preparation, and Pocket generation controls while delegating bundle assets and ONNX execution to OnnxVoice.

Document parsing, SSMD, written-to-spoken semantic preparation, logical voice binding, semantic pauses, markers, timeline composition, and final mastering are intentionally outside PocketSynth.

## Install

Install PocketSynth with the OnnxVoice CPU runtime:

```bash
python -m pip install 'pocketsynth[cpu]'
```

The `gpu` extra selects OnnxVoice's GPU runtime. `playback` adds `sounddevice` for `RenderedSegment.play()`.

## Direct engine use

Use `PocketRuntime` for a reusable local or managed bundle session. Prepare a voice once and reuse it across independent requests:

```python
from pocketsynth import PocketRuntime, SynthesisSegment

with PocketRuntime.from_pretrained("english_2026-04") as runtime:
    voice = runtime.prepare_voice("alba")
    result = runtime.synthesize(
        SynthesisSegment(
            id="line-001",
            text="Hello from Pocket.",
            language="en",
        ),
        voice=voice,
    )
    result.save_wav("hello.wav")
```

For standalone text synthesis, `synthesize_text()` creates the request and prepares non-`PreparedVoice` inputs:

```python
with PocketRuntime.from_pretrained("english_2026-04") as runtime:
    result = runtime.synthesize_text("Hello from Pocket.", voice="alba")
    result.save_wav("hello.wav")
```

`PocketRuntime.load(directory)` opens a local bundle without catalog access or network activity. `PocketRuntime.from_pretrained(bundle)` resolves and opens a managed bundle through OnnxVoice.

The convenience function is useful for one-off managed output:

```python
from pocketsynth import synthesize_to_wav

synthesize_to_wav(
    "Hello from Pocket.",
    "hello.wav",
    bundle="english_2026-04",
    voice="alba",
)
```

`synthesize()` returns a `RenderedSegment` in memory. Results contain native-rate mono float32 audio, caller request identity, resolved language, token IDs, optional request-local model chunks, runtime diagnostics, and engine timing. WAV saving writes mono PCM16 and clips only during PCM conversion.

## Pocket text and model chunks

Input text must already be speakable. PocketSynth does not parse documents, expand numbers or dates, or apply pronunciation directives. The ergonomic plain-text APIs `PocketRuntime.synthesize_text()`, `synthesize()`, and `synthesize_to_wav()` split prose into sentences with Phrasplit's lightweight regex backend by default, then apply Pocket's model token limit to each sentence. This sentence segmentation is not document planning.

Pass `sentence_split="none"` to bypass sentence segmentation and use only Pocket model-limit chunking. The low-level `PocketRuntime.synthesize(SynthesisSegment(...))` keeps that model-limit-only behavior. `PocketFrontend` owns Pocket-specific whitespace normalization, configured semicolon replacement and short-input padding, SentencePiece encoding, and token-limit subdivision.

```python
with PocketRuntime.from_pretrained("english_2026-04") as runtime:
    long_text = "Dr. Smith arrived early. Then he started the presentation."
    result = runtime.synthesize_text(long_text, voice="alba")
    already_segmented = runtime.synthesize_text(
        long_text, voice="alba", sentence_split="none"
    )
```

Model chunks are request-local inference details. Their audio is joined in order with no document pauses or timeline composition. `iter_chunks()` yields those model chunks when incremental consumption is useful.

An explicit request language is a compatibility assertion against the active bundle. `None` uses the bundle language. PocketSynth does not switch bundles or route languages automatically.

## Voices

Pocket bundles and voice conditioning are separate. `prepare_voice()` accepts bundle-declared predefined names, mono PCM16 WAV paths, `Path` objects, in-memory `(audio, sample_rate)` tuples, and existing `PreparedVoice` objects. Reference audio is validated and resampled to the bundle rate before OnnxVoice encodes it.

`PreparedVoice` can be reused for many requests on a compatible runtime. Its fingerprint identifies canonical resampled reference audio or the bundle and predefined voice name. OnnxVoice remains responsible for voice-state assets, downloads, authentication, and cache integrity. Some predefined voice assets require accepting upstream access terms and authenticating with Hugging Face before online use.

To inspect dependencies, providers, bundle metadata, and voice format:

```bash
pocketsynth check --bundle english_2026-04 --voice alba
pocketsynth check --bundle-dir ./onnx/english_2026-04 --voice reference.wav
```

## CLI

Managed bundle:

```bash
pocketsynth synthesize \
  --bundle english_2026-04 \
  --voice alba \
  --output hello.wav \
  "Hello from Pocket."
```

Local bundle:

```bash
pocketsynth synthesize \
  --bundle-dir ./onnx/english_2026-04 \
  --voice reference.wav \
  --output hello-local.wav \
  "Hello from Pocket."
```

The `synthesize` command accepts `--sentence-split phrasplit` (the default lightweight regex backend, without spaCy) or `--sentence-split none` to disable sentence segmentation and apply only Pocket model-limit chunking. Other inference controls include `--temperature`, `--lsd-steps`, `--max-frames`, and `--frames-after-eos`. Asset controls include `--cache-dir`, `--offline`, `--refresh-catalog`, and `--force-download` for managed bundles.

## Orchestrated use

An application that renders documents resolves document meaning and concrete engine inputs before calling PocketSynth:

```text
source document / SSMD
    -> application orchestration and document planning
    -> prepared speakable text + concrete bundle + PreparedVoice
    -> PocketRuntime
    -> RenderedSegment
    -> application composition and output policy
```

PocketSynth does not import or require the neighboring document-planning or audio-composition packages. Callers own semantic pauses, markers, logical voice roles, external audio, timeline placement, and final mastering.

## Breaking boundary change

The next breaking release removes the former document-planning and composition surface. There are no compatibility aliases. Replace `PocketPipeline` plan/render methods with `PocketRuntime.synthesize()` or `synthesize_text()`. Pass prepared speakable text and a concrete voice. Build document pauses, markers, role routing, and final timelines in the application or its composition layer.

The change also removes planner configuration and diagnostics, plan provenance, semantic-unit streaming, AudioJob creation, and core output gain/normalization settings. Pocket model chunks remain supported as request-local chunks.

## Examples and checks

See [`examples/README.md`](examples/README.md). The runner does not start managed downloads unless requested:

```bash
python examples/run_all.py --list
python examples/run_all.py --local-only
python examples/run_all.py --include-network
python examples/run_all.py --include-network --offline
```

Set `POCKETSYNTH_EXAMPLE_VOICE` for reference-WAV examples and `POCKETSYNTH_EXAMPLE_BUNDLE_DIR` for local examples.

Run project checks with:

```bash
python -m pytest
python -m ruff check pocketsynth tests examples
python -m mypy --config-file pyproject.toml pocketsynth
python -m compileall -q pocketsynth tests examples
```

Real model tests are environment-gated. Set `POCKETSYNTH_TEST_BUNDLE_DIR` and `POCKETSYNTH_TEST_VOICE_WAV` to run the local integration tests. Managed tests may require network access and accepted upstream voice-asset terms.

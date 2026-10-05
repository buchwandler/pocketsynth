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

Managed Kyutai prompt support uses the OnnxVoice prompt catalog and cache API. The supported dependency range remains `onnxvoice>=0.2.2,<0.3`; installation extras select the desired runtime backend.

## Direct engine use

Use `PocketRuntime` for a reusable local or managed bundle session. Prepare a voice once and reuse it across independent requests:

```python
from pocketsynth import PocketRuntime, SynthesisRequest

with PocketRuntime.from_pretrained("english_2026-04") as runtime:
    request = SynthesisRequest(
        id="line-001",
        text="Hello from Pocket.",
        language="en",
    )
    measure = runtime.measure_request(request)
    print(measure.amount, measure.maximum, measure.fits)
    voice = runtime.prepare_voice("alba")
    result = runtime.synthesize(request, voice=voice)
    result.save_wav("example-artifacts/hello.wav")
```

`PocketRuntime.synthesize()` is strict and atomic. It encodes the complete request once, then either performs one inference or raises `SynthesisInputTooLongError`. It never splits text. `config=` accepts a validated `GenerationConfig`; `synthesize_text()` is a strict plain-text wrapper that prepares non-`PreparedVoice` inputs.

`runtime.measure_request(request)` validates and encodes the complete request without preparing a voice, fetching assets, inferring, or splitting text. It returns a public `RequestMeasure` with `amount`, `maximum`, `unit` (`"tokens"`), and `fits`; an oversized measurement reports `fits=False`, while `synthesize()` still raises `SynthesisInputTooLongError` before inference. Measurement and synthesis share their request encoding path, so callers do not need to inspect `runtime.frontend` or `runtime.metadata`.

`request_api_contract()` returns the versioned public capability declaration (`REQUEST_API_VERSION == 1`), including caller-owned text boundaries and the features supported by this engine. Creating the descriptor does not inspect assets or open a runtime. Import `AssetError`, `AssetDownloadError`, `AssetAccessError`, `AssetCacheError`, `CatalogUnavailableError`, `OfflineAssetError`, `SessionCreationError`, and `RuntimeCapabilityError` from `pocketsynth` for typed handling of public asset/runtime failures.

`PocketRuntime.load(directory)` opens a local bundle without catalog access or network activity. `PocketRuntime.from_pretrained(bundle)` resolves and opens a managed bundle through OnnxVoice.

Document-friendly splitting is available only from the explicit convenience module:

```python
from pocketsynth.convenience import synthesize_to_wav

synthesize_to_wav(
    "Hello from Pocket.",
    "example-artifacts/hello-convenience.wav",
    bundle="english_2026-04",
    voice="alba",
    sentence_split="phrasplit",  # opt in to sentence segmentation
)
```

`pocketsynth.convenience.synthesize_with_runtime()` and `pocketsynth.convenience.synthesize()` return a `RenderedSegment`. `pocketsynth.convenience.synthesize_to_wav()` returns the destination `Path` after atomically writing the WAV. Convenience rendering may split at Pocket model limits. Sentence splitting defaults to `"none"`; pass `sentence_split="phrasplit"` to opt in. Convenience functions are not aliases for the strict runtime API and are not imported by `pocketsynth`.

## Text and model chunks

Input text must already be speakable. PocketSynth does not parse documents, expand numbers or dates, or apply pronunciation directives. Strict synthesis neither segments sentences nor subdivides at the model token limit. Oversized requests fail with `SynthesisInputTooLongError`.

The explicit convenience layer can split text to fit model limits. Its optional Phrasplit mode segments sentences first, then Pocket model-limit chunks are rendered and joined in order. `PocketRuntime.iter_chunks()` exposes model-limit chunks for applications that explicitly want chunked rendering; it does not perform sentence segmentation.

```python
from pocketsynth.convenience import synthesize_with_runtime
from pocketsynth import PocketRuntime

with PocketRuntime.from_pretrained("english_2026-04") as runtime:
    long_text = "Dr. Smith arrived early. Then he started the presentation."
    result = synthesize_with_runtime(runtime, long_text, voice="alba", sentence_split="phrasplit")
```

An explicit request language is a compatibility assertion against the active bundle. `None` uses the bundle language. PocketSynth does not switch bundles or route languages automatically.

## Voices

Pocket bundles and voice conditioning are separate. `prepare_voice()` accepts bundle-declared predefined names, local PCM WAV paths, `Path` objects, in-memory `(audio, sample_rate)` tuples, existing `PreparedVoice` objects, and managed `kyutai-tts-voices:<id>` references. Reference WAVs support uncompressed PCM 8-, 16-, 24-, and 32-bit mono or multichannel audio; multichannel input is downmixed and audio is resampled to the bundle rate before OnnxVoice encodes it.

`PreparedVoice` can be reused for many requests on a compatible runtime. For a local reference, `fingerprint` identifies normalized audio and sample rate. For a managed prompt, `voice_prompt` carries typed catalog provenance, while `bundle_revision` identifies the model bundle. A managed identity keeps the catalog asset SHA-256 (`source_sha256`) separate from the normalized prepared-audio fingerprint (`prepared_sha256`).

OnnxVoice 0.2.2 owns prompt catalogs, prompt WAV fetching, cache integrity, and offline behavior. PocketSynth delegates managed audio fetching to OnnxVoice during `prepare_voice()`; metadata discovery does not fetch audio or open a model. Predefined voice states are separate assets and may require accepting upstream access terms and authenticating with Hugging Face.

The simple managed workflow uses the current catalog identity for a reference:

```python
from pocketsynth import PocketRuntime

with PocketRuntime.from_pretrained("english_2026-04") as runtime:
    voice = runtime.prepare_voice("kyutai-tts-voices:alba-mackenna/casual")
    result = runtime.synthesize_text("Hello from a managed Kyutai voice.", voice=voice)
    result.save_wav("example-artifacts/kyutai-casual.wav")
```

For reproducible orchestration, inspect and retain a `VoicePromptInfo` before opening the runtime. Inspection is metadata-only. Passing that record to `prepare_voice()` checks its prompt reference, catalog SHA-256, and source revision against the current catalog before fetching audio; a changed identity raises `VoicePromptChangedError`.

```python
from pocketsynth import PocketRuntime, inspect_voice_prompt

prompt = inspect_voice_prompt("kyutai-tts-voices:alba-mackenna/casual")
# Persist prompt.ref, prompt.sha256, and prompt.source_revision with the render plan.

with PocketRuntime.from_pretrained("english_2026-04") as runtime:
    voice = runtime.prepare_voice(prompt)
    assert voice.voice_prompt == prompt
    identity = voice.identity
    # identity['source_sha256'] is the catalog WAV hash.
    # identity['prepared_sha256'] is the normalized audio fingerprint.
```

`list_voice_prompts()` and `inspect_voice_prompt()` return typed prompt metadata without requiring a runtime, installing a bundle, or fetching WAVs. `discover_bundles()` similarly returns PocketSynth-owned `DiscoveredBundle` and `DiscoveredVoice` records without installing or opening a bundle; the DTOs validate catalog fields, and `DiscoveredBundle.metadata` is a copied read-only mapping. `runtime_identity(bundle)` reports `engine_version`, `runtime_revision`, `request_api_version`, `catalog_revision`, and `bundle_revision` without opening a model; OnnxVoice 0.2.2 does not expose a separate catalog revision, so `catalog_revision` is `None`. The discovery APIs accept explicit `offline` and `refresh` options; `runtime_identity()` only reads installed package metadata. The runtime list/inspect wrappers remain available, but standalone functions are the metadata-discovery entry points.

The CLI supports metadata-only prompt listing and direct prompt inspection. Filters include `--dataset`, `--variant`, `--license`, and `--offline`:

```bash
pocketsynth voices list --dataset alba-mackenna
pocketsynth check --offline --voice kyutai-tts-voices:alba-mackenna/casual
```

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
  --output example-artifacts/hello.wav \
  "Hello from Pocket."
```

Managed Kyutai reference:

```bash
pocketsynth synthesize \
  --bundle english_2026-04 \
  --voice kyutai-tts-voices:alba-mackenna/casual \
  --output example-artifacts/kyutai-casual.wav \
  "Hello from PocketSynth."
```

After the bundle and prompt are cached, pass `--offline` to use them without network access. List available catalog prompts without downloading WAVs:

```bash
pocketsynth voices list --dataset alba-mackenna
```

Local bundle:

```bash
pocketsynth synthesize \
  --bundle-dir ./onnx/english_2026-04 \
  --voice reference.wav \
  --output example-artifacts/hello-local.wav \
  "Hello from Pocket."
```

The `synthesize` CLI is a convenience renderer. Sentence splitting defaults to `none`, which still permits Pocket model-limit chunking. Pass `--sentence-split phrasplit` to opt into Phrasplit's lightweight regex sentence segmentation. Other inference controls include `--temperature`, `--lsd-steps`, `--max-frames`, and `--frames-after-eos`. Asset controls include `--cache-dir`, `--offline`, `--refresh-catalog`, and `--force-download` for managed bundles.

## Orchestrated use

An application that renders documents resolves document meaning and concrete engine inputs before calling PocketSynth:

```text
source document / SSMD
    -> application orchestration and document planning
    -> prepared speakable text + concrete bundle + PreparedVoice
    -> PocketRuntime
    -> SynthesisResult
    -> application composition and output policy
```

PocketSynth does not import or require the neighboring document-planning or audio-composition packages. Callers own semantic pauses, markers, logical voice roles, external audio, timeline placement, and final mastering.

## Breaking boundary change

PocketSynth's engine API is strict and atomic: `PocketRuntime.synthesize()` accepts one `SynthesisRequest` and never splits it. `synthesize_text()` is the strict plain-text wrapper. Document and model-limit splitting lives in the explicit `pocketsynth.convenience` module and the CLI. Convenience imports are not loaded by importing `pocketsynth`.

The strict API returns finite mono float32 audio with request and runtime metadata, but no chunk collection. Applications that deliberately need model-limit chunks can use `PocketRuntime.iter_chunks()` or the explicit convenience layer. Document pauses, markers, role routing, and final timelines remain application responsibilities.

### Migration from 0.1

| 0.1 API or responsibility                                     | 0.2 replacement                                                                                                                                                       |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PocketPipeline` and `set_default_voice()`                    | Open with `PocketRuntime.load()` or `from_pretrained()`, prepare a voice, and pass it to each synthesis call                                                          |
| UtterPlan, document parsing, and semantic preparation         | Prepare speakable text in the application before calling PocketSynth                                                                                                  |
| Implicit sentence/document splitting                          | Use `pocketsynth.convenience.synthesize_with_runtime()` or convenience `synthesize()` with `sentence_split="phrasplit"`; use `"none"` to bypass sentence segmentation |
| Pipeline document-unit streaming                              | Use `PocketRuntime.iter_chunks()` only for explicit model-limit chunks; application code owns semantic units                                                          |
| `AudioResult`, AudioJob, and AudioCompose timelines           | Use strict `SynthesisResult` or convenience `RenderedSegment`; compose timelines in the application                                                                   |
| Package-root `synthesize()` and `synthesize_to_wav()` imports | Import these functions from `pocketsynth.convenience`                                                                                                                 |
| `normalize_audio` and `volume` request controls               | Use static `VoiceLevelConfig` calibration where appropriate, or perform mastering in the application                                                                  |

## Examples and checks

See [`examples/README.md`](examples/README.md). The runner does not start managed downloads unless requested:

```bash
python examples/run_all.py --list
python examples/run_all.py --local-only
python examples/run_all.py --include-network
python examples/run_all.py --include-network --offline
```

`examples/kyutai_voice.py` uses a managed catalog reference and needs no local WAV or `POCKETSYNTH_EXAMPLE_VOICE`. Set `POCKETSYNTH_EXAMPLE_VOICE` for local reference-WAV examples and `POCKETSYNTH_EXAMPLE_BUNDLE_DIR` for local bundles. `examples/clone_all_kyutai_voices.py` discovers and renders all cataloged WAV prompts; see the examples guide for filters and dry-run usage.

Run project checks with:

```bash
python -m pytest
python -m ruff check pocketsynth tests examples
python -m mypy --config-file pyproject.toml pocketsynth
python -m compileall -q pocketsynth tests examples
```

Real model tests are environment-gated. Set `POCKETSYNTH_TEST_BUNDLE_DIR` and `POCKETSYNTH_TEST_VOICE_WAV` to run the local integration tests. To opt into one pinned Kyutai prompt download and synthesis followed by offline cache reuse, set `POCKETSYNTH_TEST_MANAGED_PROMPT=1`; the test uses `english_2026-04` unless `POCKETSYNTH_TEST_BUNDLE` overrides it. It does not synthesize the full catalog. Managed tests may require network access and accepted upstream voice-asset terms.

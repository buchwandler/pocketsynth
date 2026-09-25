# PocketSynth and OnnxVoice contract

This document describes the current integration boundary. PocketSynth owns a small synthesis engine API for already-prepared speakable text. OnnxVoice owns Pocket model and voice-state assets and ONNX execution. Applications own document semantics and final audio composition.

```text
source document / SSMD
        -> application orchestration and speakable-text preparation
        -> PocketRuntime strict request API
             or explicit convenience sentence/model chunking
        -> OnnxVoice Pocket runtime
        -> SynthesisResult or RenderedSegment
        -> caller-owned composition and output policy
```

## Ownership

PocketSynth owns:

- local and managed runtime lifecycle;
- Pocket text normalization and SentencePiece encoding;
- one-request validation and model-capacity checks;
- concrete `PreparedVoice` wrappers and Pocket generation controls;
- explicit model-limit chunk iteration;
- optional convenience sentence segmentation and chunk joining;
- waveform validation, diagnostics, and WAV conversion.

OnnxVoice owns:

- the Pocket bundle catalog, aliases, installation, cache, and integrity checks;
- providers, ONNX sessions, graph contracts, and inference;
- predefined voice-state lookup, downloads, credentials, and cache integrity;
- reference-voice encoding, Pocket decoding, and native sample rates.

The application owns document parsing, SSMD handling, written-to-spoken preparation, logical voice roles, semantic pauses, markers, timelines, external audio, and final mastering. PocketSynth does not depend on UtterPlan, AudioCompose, or SSMD.

## Compatible OnnxVoice version

PocketSynth requires `onnxvoice>=0.1.12,<0.2`. The CPU and GPU extras use the same supported version range and request OnnxVoice's `pocket` support. OnnxVoice 0.1.12 or newer in this range provides the Pocket runtime and local bundle support used here.

## Runtime lifecycle and voices

`PocketRuntime.load(directory)` opens a concrete local bundle without catalog access or network activity. `PocketRuntime.from_pretrained(bundle)` resolves and opens a managed bundle through OnnxVoice.

`runtime.prepare_voice(source)` returns a reusable `PreparedVoice`. Sources include a bundle-declared predefined voice name, a mono PCM16 reference WAV, in-memory audio with a sample rate, or a compatible prepared voice. PocketSynth validates and prepares reference audio; OnnxVoice encodes and resolves voice state. A declared predefined voice is not a promise that its separate asset is public or already cached. Some assets require accepted upstream terms and authenticated Hugging Face access.

```python
from pocketsynth import PocketRuntime, SynthesisRequest

with PocketRuntime.from_pretrained("english_2026-04") as runtime:
    voice = runtime.prepare_voice("alba")
    result = runtime.synthesize(
        SynthesisRequest(id="line-001", text="Hello from Pocket.", language="en"),
        voice=voice,
    )
    result.save_wav("hello.wav")
```

## Strict runtime API

`PocketRuntime.synthesize(request, voice=..., config=..., voice_level=...)` validates a complete `SynthesisRequest`, encodes its text once, checks model capacity once, and performs at most one inference. It returns a finite mono float32 `SynthesisResult`. It never performs sentence splitting or silently divides an oversized request; over-capacity text raises `SynthesisInputTooLongError`.

`PocketRuntime.synthesize_text(text, voice=...)` is a strict plain-text wrapper. It prepares a non-`PreparedVoice` source and submits one request. `PocketRuntime.iter_chunks(segment, voice=...)` is a separate explicit operation that divides at the model token limit and yields `RenderedChunk` values. It does not run sentence segmentation.

## Convenience rendering

The convenience API is an explicit module import, not a package-root export:

```python
from pocketsynth import PocketRuntime
from pocketsynth.convenience import synthesize_with_runtime

with PocketRuntime.from_pretrained("english_2026-04") as runtime:
    rendered = synthesize_with_runtime(
        runtime,
        "Dr. Smith arrived early. Then he started the presentation.",
        voice="alba",
        sentence_split="phrasplit",
    )
```

`sentence_split="phrasplit"` opts into Phrasplit's lightweight regex backend with spaCy disabled. `sentence_split="none"` bypasses Phrasplit entirely. Both modes can still use Pocket model-limit chunking. The convenience path prepares the voice once, preserves chunk order, and joins waveform chunks into a `RenderedSegment`.

The managed convenience functions `synthesize()` and `synthesize_to_wav()` are also in `pocketsynth.convenience`. The former returns a `RenderedSegment`; the latter writes a mono PCM16 WAV atomically and returns the destination `Path`. Sentence splitting defaults to `none` for these functions and for the CLI.

## Real-model checks

The integration tests are gated by assets and access. They verify positive, finite, non-silent audio and WAV format.

### Local bundle and reference WAV

```bash
export POCKETSYNTH_TEST_BUNDLE_DIR=/path/to/onnx/english_2026-04
export POCKETSYNTH_TEST_VOICE_WAV=/path/to/reference.wav
python -m pytest -q tests/integration/test_real_local_wav.py
python -m pytest -q tests/test_first_wav_integration.py
```

### Managed bundle and reference WAV

```bash
export POCKETSYNTH_TEST_VOICE_WAV=/path/to/reference.wav
python -m pytest -q -m 'integration and network' tests/integration/test_real_managed_wav.py
```

`POCKETSYNTH_TEST_BUNDLE` may select a managed bundle; otherwise the test uses `english_2026-04`. It checks both online synthesis and reuse from the populated cache in offline mode.

### Managed predefined voice

After configuring gated Hugging Face access and credentials:

```bash
export POCKETSYNTH_TEST_PREDEFINED_VOICE=1
python -m pytest -q -m 'integration and network' tests/integration/test_real_predefined_voice_wav.py
```

### Long-text modes

With managed assets and access configured, run `python examples/long_text.py`. It writes one WAV using `sentence_split="phrasplit"` and another using `sentence_split="none"`. Check that both are mono PCM16, at the bundle sample rate, positive-duration, finite, and non-silent. The no-split path bypasses Phrasplit while retaining Pocket model-limit chunking. Listening to short outputs is useful in addition to checking their WAV structure.

## 0.1 migration note

There is no current `PocketPipeline`, `set_default_voice()`, UtterPlan integration, AudioCompose ownership, or package-root convenience export. Replace the old document-oriented pipeline with application-owned preparation and `PocketRuntime`; pass a concrete voice on each request. Use the explicit convenience module only when sentence/model chunking is desired. Applications that need document units, pauses, markers, or timelines must implement those responsibilities outside PocketSynth.

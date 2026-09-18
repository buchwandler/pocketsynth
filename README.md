# pocketsynth

`pocketsynth` is an UtterPlan-aware Python runtime for Pocket TTS ONNX bundles. This MVP is shaped
from the current PiperSynth architecture: it treats OnnxVoice as the sole model catalog/store/ORT
boundary and keeps semantic planning in UtterPlan.

It intentionally has **no `src/` layer**. The import package lives at repository root.

## Status

This MVP targets an OnnxVoice version that implements the accompanying Pocket support brief. The
supplied current OnnxVoice snapshot does not yet register a `pocket` system, parse the Pocket bundle
catalog, or expose generic local multi-file opening; those are the required companion changes.

## Installation

```bash
pip install -e '.[cpu]'
```

Versioning is dynamic through `setuptools-scm`; Git tags are the source of the package version and
`pocketsynth/_version.py` is generated during build/editable installation.

## Managed bundle

```python
from pocketsynth import PocketPipeline

with PocketPipeline.from_pretrained(
    "english_2026-04",
    precision="int8",
) as tts:
    tts.set_default_voice("reference.wav")  # mono 16-bit PCM WAV in MVP
    result = tts("Hello from Pocket.")
    result.save_wav("hello.wav")
```

`from_pretrained()` delegates installation to OnnxVoice. Pocketsynth never downloads the model
bundle itself.

## Strictly local bundle

```python
from pocketsynth import PocketPipeline

with PocketPipeline.load(
    "./onnx/english_2026-04",
    precision="int8",
) as tts:
    tts.set_default_voice("reference.wav")
    tts("Hello.").save_wav("hello.wav")
```

`load()` is network-free and resolves concrete files from the directory before calling the proposed
`onnxvoice.open_local(system="pocket", files=...)` API.

## UtterPlan behavior

Sentence units are the default:

```python
plan = tts.plan(document, unit="sentence")
result = tts.render_plan(plan, voice="reference.wav")
```

The adapter keeps UtterPlan's semantic boundaries authoritative. Compatible adjacent segments are
merged inside a sentence; resolved pauses, language changes, or voice changes split render spans.
Only after that does the Pocket frontend apply `max_token_per_chunk` subdivision using the bundle's
SentencePiece tokenizer.

So the hierarchy is:

```text
UtterPlan sentence unit
  -> one or more semantic Pocket render spans
      -> one or more tokenizer-limited Pocket model chunks
```

## Voice reuse

```python
with PocketPipeline.from_pretrained("english_2026-04") as tts:
    narrator = tts.prepare_voice("narrator.wav")
    tts.set_default_voice(narrator)
    one = tts("First sentence.")
    two = tts("Second sentence.")
```

The Mimi encoder runs once for the prepared voice; the returned state is reusable.

UtterPlan logical voice directives can be mapped explicitly:

```python
result = tts.render_plan(
    plan,
    voice=narrator,
    voice_bindings={"alice": tts.prepare_voice("alice.wav")},
)
```

This MVP does not silently resolve named Pocket predefined voices because those state files are a
separate upstream/gated asset source. That should be added to OnnxVoice as an explicit asset family.

## Runtime ownership

Pocketsynth owns:

- UtterPlan adaptation and policy;
- SentencePiece frontend and model-limit splitting;
- voice binding / prepared voice reuse;
- Pocket generation controls (`temperature`, `lsd_steps`, frame limits);
- AudioCompose timeline construction and application-facing result types.

OnnxVoice owns:

- `pocket-onnx-bundles` catalog parsing;
- download/cache/checksum/install manifests;
- ORT providers/session options;
- the coordinated Pocket sessions;
- state-manifest arrays;
- Mimi voice encoding;
- Flow-LM + flow-matching loop;
- Mimi decoding;
- canonical native-rate inference output.

See `docs/architecture.md` and the separate `onnxvoice-pocket-support-implementation-brief.md`.

## CLI

```bash
pocketsynth synthesize \
  --bundle english_2026-04 \
  --voice reference.wav \
  --output hello.wav \
  "Hello world."
```

or local-only:

```bash
pocketsynth synthesize \
  --bundle-dir ./onnx/english_2026-04 \
  --voice reference.wav \
  --output hello.wav \
  "Hello world."
```

## MVP limitations

- Pocket support must first be added to OnnxVoice per the implementation brief.
- Reference WAV input is mono 16-bit PCM in this MVP.
- Named upstream `.safetensors` voice states are not downloaded.
- Streaming currently operates at UtterPlan sentence-unit granularity; the OnnxVoice brief reserves
  a native `PocketAdapter.stream()` capability for a later Pocketsynth streaming path.
- The bootstrap `pocket-onnx-bundles` catalog should be refreshed once OnnxVoice's builder exists so
  current Hugging Face sizes and SHA-256 values are materialized automatically.

# Pocket runtime contract

This document records the current boundary between PocketSynth and OnnxVoice. The former implementation brief described Pocket support as missing. Pocket catalog registration, managed installation, local multi-file opening, and the v2 runtime contract are now companion OnnxVoice responsibilities.

## Ownership

```text
pocket-onnx-bundles catalog
        -> OnnxVoice catalog, cache, installation, and runtime
        -> PocketSynth frontend, policy, voice lifecycle, composition, and WAV API
```

PocketSynth owns:

- UtterPlan planning and adaptation;
- SentencePiece tokenization and Pocket text normalization;
- token-limit subdivision before runtime calls;
- logical voice bindings and reusable `PreparedVoice` wrappers;
- generation configuration;
- AudioCompose timeline construction and result types;
- examples, CLI behavior, and application-facing progress events.

OnnxVoice owns:

- the Pocket catalog and bundle aliases;
- download, cache, checksum, and installation manifests;
- ORT providers, sessions, and session options;
- bundle graph metadata and state manifests;
- Mimi voice encoding and decoding;
- Flow-LM and flow matching;
- canonical native-rate inference output.

PocketSynth must not parse graph input/output names, state manifests, Flow-LM cache layout, or Mimi decoder state. It may consume high-level bundle metadata such as language, sample rate, token limit, text normalization flags, and recommended frames after EOS.

## Minimum compatible runtime

PocketSynth declares `onnxvoice>=0.1.10,<0.2`. The compatible runtime exposes:

```python
installation = onnxvoice.OnnxVoice().install(
    "pocket:english_2026-04",
    quality="int8",
)

with onnxvoice.OnnxVoice().open(installation) as runtime:
    voice = runtime.prepare_voice(reference_audio, sample_rate=24000)
    result = runtime.infer(
        token_ids,
        voice_state=voice,
        temperature=0.7,
        lsd_steps=1,
    )
```

`PocketVoiceState` is an OnnxVoice implementation detail. PocketSynth exposes it through the stable `PreparedVoice` wrapper, so application code does not depend on whether the state is an embedding object, mapping, or another reusable representation.

## Managed and local opening

Managed use resolves and installs through OnnxVoice:

```python
with PocketPipeline.from_pretrained(
    "english_2026-04",
    precision="int8",
) as pipeline:
    pipeline.set_default_voice("reference.wav")
    result = pipeline("Hello from Pocket.")
```

Local use is network-free and passes semantic component files to OnnxVoice:

```python
with PocketPipeline.load("./onnx/english_2026-04", precision="int8") as pipeline:
    pipeline.set_default_voice("reference.wav")
    result = pipeline("Hello from Pocket.")
```

The local bundle contains `bundle.json`, tokenizer and conditioning files, and the selected ONNX components. PocketSynth locates those files but does not open their graphs itself.

## Voice and token contracts

The application boundary accepts mono 16-bit PCM WAV prompts. PocketSynth reports the actual channels, sample width, sample rate, and compression when that contract is violated, and resamples valid prompts to the bundle rate.

UtterPlan semantic units become Pocket render spans. The Pocket frontend then applies SentencePiece and splits model calls so every token array is at most `max_token_per_chunk`. OnnxVoice still rejects an oversized array as a defensive runtime contract check. OnnxVoice does not own text semantics or silently split text.

A prepared voice is reusable:

```python
voice = pipeline.prepare_voice("reference.wav")
pipeline.set_default_voice(voice)
pipeline("First sentence.")
pipeline("Second sentence.")
```

The Mimi encoder is called once for the prepared voice. Named predefined voice assets remain outside this package's managed bundle flow.

## Progress contract

OnnxVoice emits asset events for catalog, download, verification, and installation. PocketSynth maps those events to `AssetProgressEvent`, `AssetProgressCallback`, and `ConsoleAssetProgress` without exposing OnnxVoice event classes:

```python
from pocketsynth import ConsoleAssetProgress, PocketPipeline

with PocketPipeline.from_pretrained(
    "english_2026-04",
    progress=ConsoleAssetProgress(),
) as pipeline:
    ...
```

## Acceptance checks

The real local test uses:

```bash
export POCKETSYNTH_TEST_BUNDLE_DIR=/path/to/onnx/english_2026-04
export POCKETSYNTH_TEST_VOICE_WAV=/path/to/reference.wav
pytest -q -m integration tests/integration/test_real_local_wav.py
```

The managed test is network-marked and verifies catalog resolution, installation, inference, WAV format, cached reuse, and explicit offline reuse:

```bash
export POCKETSYNTH_TEST_VOICE_WAV=/path/to/reference.wav
pytest -q -m 'integration and network' tests/integration/test_real_managed_wav.py
```

Both checks require positive frame count, finite non-silent audio, mono channels, 16-bit PCM, and the bundle sample rate.

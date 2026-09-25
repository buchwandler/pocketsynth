# Architecture

PocketSynth is an engine for Pocket TTS ONNX bundles. Its strict runtime API renders one complete `SynthesisRequest` with one encode, one capacity decision, and at most one inference. It does not segment or split a strict request. Explicit document and model-limit convenience splitting is available from `pocketsynth.convenience` and the CLI, outside the strict package import surface.

```text
source document / SSMD
        |
        v
application orchestration and document preparation
        |
        | one prepared SynthesisRequest + PreparedVoice
        v
+-----------------------------------------------+
|                  PocketSynth                  |
|                                               |
| strict request validation                    |
| Pocket text normalization and encoding       |
| one model-capacity check                     |
| voice preparation and calibration metadata   |
| OnnxVoice Pocket inference                   |
| finite mono float32 SynthesisResult          |
+----------------------+------------------------+
                       |
                       v
                caller / application
                       |
                       v
              application composition

Explicit convenience path:
  sentence splitting (opt-in) -> model-limit chunks -> joined RenderedSegment
```

## Ownership

1. **Application orchestration and document planning** own source parsing, SSMD, written-to-spoken preparation, semantic units, language routing, logical voice roles, directives, pauses, and marker resolution.
2. **PocketSynth strict runtime** owns bundle/runtime lifecycle, request validation, Pocket-specific model text normalization, SentencePiece encoding, one-request capacity validation, concrete voice preparation, generation controls, static voice-level calibration, waveform validation, diagnostics, and `SynthesisResult` construction.
3. **PocketSynth convenience module and CLI** may segment sentences and divide text at model limits, then join the rendered chunks into a `RenderedSegment`. Sentence splitting defaults to `none`; Phrasplit is opt-in. These modules are not imported by `pocketsynth` or `pocketsynth.runtime`.
4. **OnnxVoice** owns Pocket catalog and asset resolution, installation and cache integrity, provider selection, ONNX sessions, graph contracts, predefined voice-state resolution, Mimi voice encoding, Flow-LM generation, flow matching, decoding, native sample rate, and runtime diagnostics.
5. **Application composition** owns explicit silence, clips, timeline position, markers, output policy, external audio, and final mastering.

PocketSynth has no runtime dependency on document-planning or audio-composition layers. Those may be used by callers around either API surface.

## Strict request and convenience rendering

`PocketRuntime.synthesize(request, voice=..., config=...)` accepts a `SynthesisRequest` and a prepared voice. It validates supported fields and compatibility before encoding the original complete text once. If the resulting token count exceeds the bundle's model capacity, it raises `SynthesisInputTooLongError`; otherwise it performs one inference and returns one `SynthesisResult`. The result has no public chunk collection. `synthesize_text()` is a strict plain-text wrapper that can prepare a supplied voice source.

```text
PocketRuntime.synthesize(SynthesisRequest(...))
    -> validate request, language, config, and voice
    -> PocketFrontend.encode(complete request text) exactly once
    -> compare token count with bundle capacity
    -> one OnnxVoice inference, or typed capacity error
    -> apply configured static voice level
    -> finite mono float32 SynthesisResult

pocketsynth.convenience.synthesize_with_runtime(...)
    -> optional sentence segmentation (Phrasplit is opt-in)
    -> PocketRuntime.iter_chunks() for model-limit subdivision
    -> ordered waveform joining
    -> RenderedSegment
```

`PocketRuntime.iter_chunks()` is an explicit chunked-rendering operation. It divides only at Pocket model limits and does not perform sentence segmentation. The convenience module owns calls to `split_text_for_synthesis`; strict runtime modules do not import it. Convenience functions return `RenderedSegment` and must not be mistaken for atomic synthesis.

## Runtime and bundle lifecycle

`PocketRuntime.load(directory)` opens a concrete local bundle without network access. `PocketRuntime.from_pretrained(bundle)` resolves and opens a managed bundle through OnnxVoice. Both expose bundle metadata, language, sample rate, predefined voices, voice preparation, token inference, diagnostics, and idempotent close/context-manager lifecycle.

An explicit request language is a compatibility assertion against the active bundle. `None` accepts the bundle-declared language. PocketSynth does not select or switch bundles.

## Voice conditioning

`PreparedVoice` is reusable conditioning state for a compatible bundle and source revision. Reference WAV data is validated as mono PCM16, resampled to the bundle rate, and encoded by OnnxVoice. Its stable identity is based on normalized accepted audio and sample rate. A predefined voice identity includes bundle ID, voice name, and source revision when available.

Voice-level configuration can leave audio unchanged, apply a catalog calibration, or explicitly override gain. Missing calibration is reported in metadata; PocketSynth does not invent measurements or measure loudness per request.

## Generation and results

`GenerationConfig` contains Pocket inference controls: temperature, LSD steps, maximum frames, and frames after EOS. When frames-after-EOS is omitted, PocketRuntime uses the bundle recommendation.

`SynthesisResult` contains finite mono float32 audio at the bundle's native sample rate, request ID and text, resolved language, empty word timings when the engine has none, and structured bundle, voice, token, generation, calibration, diagnostics, and timing metadata. WAV conversion clamps samples when writing PCM16. Output mastering remains caller-owned.

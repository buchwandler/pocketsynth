# Architecture

PocketSynth is a Pocket TTS synthesis engine. Its public session is `PocketRuntime`, and its input is already-prepared speakable text plus a resolved Pocket bundle and concrete voice conditioning state.

```text
source document / SSMD
        |
        v
application orchestration and document planning
        |
        | prepared speakable text
        | resolved Pocket bundle
        | concrete PreparedVoice
        v
+-----------------------------------------------+
|                  PocketSynth                  |
|                                               |
| PocketRuntime lifecycle                       |
| Pocket-specific text normalization             |
| SentencePiece encoding                        |
| model token-limit chunking                    |
| reference/predefined voice preparation        |
| generation controls                           |
| OnnxVoice Pocket inference                    |
| request-local chunk joining                  |
| waveform validation and WAV convenience       |
+----------------------+------------------------+
                       |
                       | independent RenderedSegment
                       v
              caller / Readio
                       |
                       v
                Audio composition
```

## Ownership

1. **Application orchestration and document planning** own source parsing, SSMD, written-to-spoken preparation, semantic units, language routing, logical voice roles, directives, semantic pauses, and marker resolution.
2. **PocketSynth** owns bundle/runtime lifecycle, Pocket-specific model text normalization, SentencePiece encoding, model token-limit chunking, concrete voice preparation, generation controls, waveform validation, request-local model-chunk joining, diagnostics, and WAV convenience.
3. **OnnxVoice** owns Pocket catalog and asset resolution, installation and cache integrity, provider selection, ONNX sessions, graph contracts, predefined voice-state resolution, Mimi voice encoding, Flow-LM generation, flow matching, decoding, native sample rate, and runtime diagnostics.
4. **Application composition** owns explicit silence, clips, timeline position, markers, resampling/output policy, external audio, and final mastering.

PocketSynth has no runtime dependency on the document planner or composition layers. They may be used by callers on either side of the engine.

## Synthesis request and model chunks

A `SynthesisSegment` is one caller-owned request. Its ID is opaque and is copied unchanged into the resulting `RenderedSegment`. PocketSynth does not interpret it as a document unit or retain plan IDs, markers, timeline offsets, semantic pauses, or logical voice roles.

```text
SynthesisSegment
    -> PocketFrontend.prepare_text()
    -> PocketFrontend.split_for_model()
    -> SentencePiece token IDs
    -> OnnxVoice inference for each token-bounded model chunk
    -> ordered request-local waveform concatenation
    -> RenderedSegment
```

Punctuation and whitespace are splitting heuristics used only when the model token limit requires subdivision. Chunk indices start at zero for each request. No synthetic document silence is inserted between chunks. `iter_chunks()` exposes the same request-local model chunks without turning them into semantic sentence units.

## Runtime and bundle lifecycle

`PocketRuntime.load(directory)` opens a concrete local bundle and is network-free. `PocketRuntime.from_pretrained(bundle)` installs or resolves a managed bundle through OnnxVoice and then opens its runtime. Both paths expose bundle metadata, language, sample rate, predefined voices, token inference, voice preparation, diagnostics, and idempotent close/context-manager lifecycle.

PocketSynth owns the compatibility assertion between an explicit request language and the active bundle language. `None` accepts the bundle-declared language. It does not select or switch bundles.

## Voice conditioning

The bundle is the acoustic/runtime target. `PreparedVoice` is a reusable conditioning state for that bundle. Reference WAV data is validated as mono PCM16, resampled to the bundle rate, converted to the canonical conditioning representation, and encoded by OnnxVoice once. Its stable fingerprint is derived from the canonical sample-rate audio. A predefined voice fingerprint contains the bundle identity and declared voice name. No asset revision is invented when OnnxVoice does not expose one.

## Generation and results

`GenerationConfig` contains only Pocket inference controls: temperature, LSD steps, maximum frames, and frames after EOS. When frames-after-EOS is omitted, PocketRuntime uses the bundle recommendation.

`RenderedSegment` contains finite mono float32 audio at the bundle's native sample rate, the caller ID and text, resolved language, flattened token IDs, request-local chunks, and engine-only diagnostics/timing. WAV conversion clamps samples when writing PCM16. Output gain, loudness, and mastering are caller-owned.

## Removed responsibility boundary

PocketSynth no longer parses SSMD, prepares semantic speech, creates plans or document units, resolves logical voice bindings, interprets directives or pause policy, stores markers, creates AudioJobs, runs a composer, or assembles document timelines. Those responsibilities remain with the caller and its selected document/audio tools.

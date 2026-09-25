# Architecture

PocketSynth is a Pocket TTS synthesis engine. `PocketRuntime.synthesize_text()` is its ergonomic plain-text entry point; it segments sentences with Phrasplit's lightweight backend by default before Pocket model-limit chunking. `PocketRuntime.synthesize(SynthesisSegment(...))` remains a low-level prepared-request API. Both use a resolved Pocket bundle and concrete voice conditioning state.

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
| Optional Phrasplit sentence segmentation      |
| Pocket-specific text normalization            |
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
2. **PocketSynth** owns bundle/runtime lifecycle, optional lightweight sentence segmentation for ergonomic text APIs, Pocket-specific model text normalization, SentencePiece encoding, model token-limit chunking, concrete voice preparation, generation controls, waveform validation, request-local model-chunk joining, diagnostics, and WAV convenience.
3. **OnnxVoice** owns Pocket catalog and asset resolution, installation and cache integrity, provider selection, ONNX sessions, graph contracts, predefined voice-state resolution, Mimi voice encoding, Flow-LM generation, flow matching, decoding, native sample rate, and runtime diagnostics.
4. **Application composition** owns explicit silence, clips, timeline position, markers, resampling/output policy, external audio, and final mastering.

PocketSynth has no runtime dependency on the document planner or composition layers. They may be used by callers on either side of the engine.

## Synthesis request and model chunks

A `SynthesisSegment` is one caller-owned low-level request. Its ID is opaque and is copied unchanged into the resulting `RenderedSegment`. PocketSynth does not interpret it as a document unit or retain plan IDs, markers, timeline offsets, semantic pauses, or logical voice roles.

The ergonomic plain-text path supports optional sentence segmentation. `synthesize_text()` defaults to Phrasplit's lightweight regex backend, forced with `use_spacy=False`; `sentence_split="none"` bypasses it. In either mode, Pocket's model splitter is the final inference safety layer. Low-level `synthesize()` and `iter_chunks()` do not perform linguistic sentence segmentation.

```text
synthesize_text(text)
    -> optional Phrasplit sentence segmentation
    -> for each sentence: PocketFrontend.prepare_text()
    -> PocketFrontend.split_for_model()
    -> SentencePiece token IDs
    -> OnnxVoice inference for each token-bounded model chunk
    -> ordered request-local waveform concatenation
    -> RenderedSegment

synthesize(SynthesisSegment(...)) / iter_chunks(SynthesisSegment(...))
    -> skip linguistic sentence segmentation
    -> PocketFrontend.prepare_text() / split_for_model()
    -> SentencePiece token IDs / OnnxVoice inference
```

Phrasplit finds linguistic boundaries; `PocketFrontend.split_for_model()` guarantees that every model request respects the token ceiling. A Phrasplit sentence is never repacked with the following sentence, though one oversized sentence can become multiple model chunks. Chunk indices start at zero for each request. No synthetic silence is inserted between chunks.

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

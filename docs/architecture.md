# Architecture

Pocketsynth follows the same dependency ownership pattern as the current PiperSynth codebase,
but its atomic model asset is a Pocket **bundle**, not a voice model.

1. **UtterPlan** owns document parsing, SSMD, written-to-spoken preparation, language runs,
   directives, pause resolution, markers, and sentence/paragraph render units.
2. **Pocketsynth frontend** owns Pocket-specific text normalization, SentencePiece tokenization,
   and model token-limit subdivision inside an UtterPlan render span.
3. **OnnxVoice** owns the Pocket catalog, shared asset store, artifact verification, provider
   selection, all Pocket ONNX sessions, state-manifest tensors, voice encoding, Flow-LM generation,
   flow matching, Mimi decoding, and raw native-rate inference results.
4. **AudioCompose** owns generic audio clips, explicit silence, final timeline composition,
   resampling/output policy, and marker anchors where exact boundaries are known.
5. **Pocketsynth** owns application policy: bundle/language compatibility, voice bindings,
   generation controls, UtterPlan adaptation, result metadata, diagnostics, and the public
   `PocketRuntime` / `PocketPipeline` APIs.

Normal batch chain:

```text
source text
  -> UtterPlan
  -> Pocketsynth plan adapter
  -> Pocket frontend / SentencePiece IDs
  -> OnnxVoice PocketAdapter
  -> raw float32 model audio
  -> Pocketsynth compatibility postprocessing
  -> AudioCompose job
  -> AudioCompose Composer
  -> Pocketsynth AudioResult
```

## Semantic units vs model chunks

UtterPlan units remain semantic/cache/streaming boundaries. The default Pocketsynth unit is a
sentence. Within one unit, compatible adjacent UtterPlan segments are merged into Pocket render
spans unless a resolved pause, language change, voice change, or unsupported directive requires a
hard boundary.

A render span may still exceed Pocket's `max_token_per_chunk`. The Pocket frontend then subdivides
that span using the bundle tokenizer. Those model chunks are an implementation detail and do not
change UtterPlan identities or units.

## Asset and runtime lifecycle

`PocketRuntime.load(directory)` is strictly local and network-free. It selects concrete local files
for the requested precision and opens them through `onnxvoice.open_local(system="pocket", files=...)`.

`PocketPipeline.from_pretrained()` installs `pocket:<bundle>` through `onnxvoice.OnnxVoice`, then
opens the returned installation. Pocketsynth does not contain a catalog parser, downloader, or
independent model cache.

The initial `int8` local profile follows current upstream runtime behavior: INT8 Flow-LM main/flow
and Mimi decoder, with FP32 Mimi encoder and text conditioner. This is deliberately conservative.

## Voice prompts

A Pocket voice is separate from the model bundle. `PreparedVoice` wraps the reusable state returned
by `PocketAdapter.prepare_voice()`. A reference WAV is decoded/resampled by Pocketsynth and passed to
OnnxVoice's Mimi encoder once, then the state can be reused across sentences.

Named predefined voice states are intentionally not downloaded by this MVP because they belong to a
separate upstream/gated asset domain. They should later be exposed by OnnxVoice as explicit assets,
not hidden network calls inside Pocketsynth.

## Lifecycle

Use context managers where possible. `close()` is idempotent. OnnxVoice owns ONNX session cleanup;
Pocketsynth owns its planner/runtime wrappers and prepared application state.

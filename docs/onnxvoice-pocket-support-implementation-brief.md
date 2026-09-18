# OnnxVoice implementation brief: Pocket ONNX bundle support

## Goal

Add first-class support for `pocket:<bundle-id>` installations and local Pocket bundle directories
without moving Pocket catalog/download/runtime responsibilities into Pocketsynth.

The target ownership model mirrors the current PiperSynth/OnnxVoice boundary:

```text
pocket-onnx-bundles   catalog data + schema + provenance + CI only
        |
        v
OnnxVoice             catalog parser + install/store + provider/session + Pocket graph runtime
        |
        v
Pocketsynth           UtterPlan adaptation + SentencePiece frontend + voice policy + composition
```

Pocketsynth should never download model assets directly and `pocket-onnx-bundles` should never
become an importable Python package.

## Current OnnxVoice baseline

The supplied OnnxVoice snapshot already has the useful generic foundation:

- `Artifact` supports arbitrary `role`, `quality`, `component`, `format`, checksum and metadata.
- `CatalogItem` and `Installation` do not require Piper-specific fields.
- `AssetStore` is generic, content-addressed, verified and system-scoped.
- `OnnxSession` already centralizes ONNX Runtime providers/session options.
- system adapters are registered through `onnxvoice.systems`.

The Pocket gaps are concentrated in four places:

1. `CatalogClient.DEFAULT_SOURCES` and `CatalogClient.list()` only know Piper and Kokoro parsers.
2. `CatalogClient.resolve(..., quality=...)` assumes quality selection means choosing a single
   artifact with role `model`; Pocket has several ONNX components and component-specific profiles.
3. `OnnxVoice.open_local()` only accepts `model/config/voices`, which cannot describe a Pocket
   bundle with five ONNX graphs plus tokenizer/metadata/BOS conditioning.
4. `SystemAdapter` is shaped around one `session` and `infer(token_ids)`, while Pocket is a
   coordinated, stateful multi-session runtime.

The CLI likewise hard-codes system choices and only exposes `catalog piper` tooling.

## Catalog contract

Add the default source:

```python
DEFAULT_SOURCES = {
    ...,
    "pocket": "https://raw.githubusercontent.com/buchwandler/pocket-onnx-bundles/main/catalog/bundles.json",
}
```

Add `_parse_pocket(data) -> list[CatalogItem]` in `onnxvoice/catalog.py`.

Recommended normalized item:

```python
CatalogItem(
    system="pocket",
    id="english_2026-04",
    kind="bundle",
    sample_rate=24000,
    aliases=("english", "en", ...),
    artifacts=(...),
    metadata={
        "language": "en",
        "layers": 6,
        "bundle_schema": 2,
        "profiles": {...},
        "source_revision": "<40-char sha>",
        "source_repository": "KevinAHM/pocket-tts-onnx",
        ...,
    },
)
```

Artifact roles should be semantic, not filename-derived:

```text
bundle_metadata
 tokenizer
bos_conditioning
flow_lm_main
flow_lm_flow
mimi_decoder
mimi_encoder
text_conditioner
```

ONNX artifacts can carry `quality="fp32"` or `quality="int8"`.

### Profile selection

Do not reuse the current `role == "model"` quality selector for Pocket. Add a system-aware
selection step, for example:

```python
def _select_quality(item: CatalogItem, quality: str | None) -> CatalogItem:
    if item.system == "pocket":
        return _select_pocket_profile(item, quality or "int8")
    ...existing behavior...
```

For Pocket, retain all unqualified/static artifacts and choose exactly one artifact for each
qualified component according to `item.metadata["profiles"][quality]`.

For the initial catalog, the recommended `int8` profile should match upstream runtime behavior:

```text
flow_lm_main      int8
flow_lm_flow      int8
mimi_decoder      int8
mimi_encoder      fp32
text_conditioner  fp32
```

A separate future profile can opt into the quantized encoder/conditioner after parity tests.

Store the selected profile in installation metadata as `selected_quality` for parity with the
existing local-only `resolve(..., quality=...)` check.

## Local multi-file opening

Generalize `OnnxVoice.open_local()` without breaking Piper/Kokoro:

```python
@staticmethod
def open_local(
    *,
    system: str,
    model: str | Path | None = None,
    config: str | Path | None = None,
    voices: str | Path | None = None,
    files: Mapping[str, str | Path] | None = None,
    artifact_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    sample_rate: int | None = None,
    ...,
): ...
```

Rules:

- old `model/config/voices` behavior remains unchanged;
- `files` maps semantic artifact roles to local paths;
- duplicate roles between legacy arguments and `files` are rejected;
- local files are never copied or registered;
- local opening remains network-free;
- local Pocket callers pass the selected concrete component files, not both precisions.

For roles that may legitimately have multiple qualities in a managed installation, the managed
manifest continues to preserve `quality`. A selected Pocket install should contain one concrete
artifact per runtime component.

## Pocket adapter

Add `onnxvoice/systems/pocket.py` with `PocketAdapter` and register it as `system = "pocket"`.

Unlike Piper/Kokoro, it should own multiple lazy sessions:

```python
self._sessions: dict[str, OnnxSession]
```

Recommended responsibilities:

- parse and validate `bundle_metadata` (`bundle.json`);
- load `bos_conditioning`;
- lazily open `flow_lm_main`, `flow_lm_flow`, `mimi_decoder`, `text_conditioner`;
- lazily open `mimi_encoder` only when voice cloning from waveform is requested;
- create state arrays from the Flow-LM and Mimi manifests in `bundle.json`;
- own the autoregressive Flow-LM loop and LSD/flow-matching loop;
- own Mimi encoding/decoding because those are graph/runtime responsibilities;
- return canonical `InferenceResult(float32 mono, sample_rate)`;
- close every created session idempotently.

### Capability methods

Keep SentencePiece outside OnnxVoice. Pocketsynth should pass model-ready token IDs.

Add Pocket-specific methods without making the base class Pocket-specific:

```python
@dataclass(frozen=True, slots=True)
class PocketVoiceState:
    values: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)

class PocketAdapter(SystemAdapter):
    def prepare_voice(self, audio: np.ndarray, *, sample_rate: int) -> PocketVoiceState: ...

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        voice_state: PocketVoiceState,
        temperature: float = 0.7,
        lsd_steps: int = 1,
        max_frames: int | None = None,
        frames_after_eos: int | None = None,
    ) -> InferenceResult: ...

    def stream(...):
        yield InferenceResult(...)
```

`SystemAdapter.session` should no longer be an abstract requirement for all systems. Replace it
with either a default capability error or a more neutral `sessions`/`close()` contract. Piper and
Kokoro can continue exposing their existing single-session property.

Do not put SentencePiece, UtterPlan, document parsing, pause policy, or audio timeline composition
in OnnxVoice.

## Pocket bundle metadata validation

At open time, reject a bundle that lacks the runtime fields required by the adapter. At minimum:

```text
schema_version
sample_rate
samples_per_frame
latent_dim
conditioning_dim
flow_lm_state_manifest
mimi_state_manifest
mimi_encoder_state_manifest (when voice cloning is used, if present in schema)
insert_bos_before_voice
bos_before_voice_file
```

The adapter should validate expected graph inputs/outputs against the manifests and fail with
`RuntimeContractError` rather than relying on array ordering accidentally matching a particular
export.

The current Pocket bundle format also exposes model-level text/runtime guidance such as
`max_token_per_chunk`, `model_recommended_frames_after_eos`, `remove_semicolons`, and
`pad_with_spaces_for_short_inputs`. Preserve those fields in installation/runtime metadata so
Pocketsynth can apply frontend/chunking policy without parsing ONNX graphs.

## Voice states

For MVP, support two voice sources:

1. waveform -> Mimi encoder -> reusable `PocketVoiceState`;
2. caller-provided `PocketVoiceState` / compatible array mapping.

Do **not** make Pocketsynth silently download predefined voice states from `kyutai/pocket-tts`.
Those assets are a separate, currently gated source domain. A later OnnxVoice extension can expose
separate refs such as `pocket-voice:<bundle>/<name>` or a secondary artifact catalog with explicit
authentication/provenance.

This keeps model-bundle installation deterministic and avoids hiding a second network dependency.

## Catalog build/verify tooling

Add `onnxvoice/catalog_tools/pocket.py` analogous to the Piper tool.

Builder algorithm:

1. Resolve `KevinAHM/pocket-tts-onnx` requested revision (default `main`) to an exact 40-char SHA.
2. Enumerate `onnx/*/bundle.json` at that revision.
3. Parse each bundle's `bundle.json` and derive bundle ID/language/runtime metadata.
4. Enumerate required static artifacts and available FP32/INT8 component variants.
5. Query Hugging Face file metadata and retain remote byte size plus Xet/LFS SHA-256 where exposed.
6. Build pinned `resolve/<exact-sha>/<path>?download=true` URLs.
7. Emit deterministic `catalog/bundles.json` and matching `catalog/source.json`.

Verifier requirements:

- exact catalog kind/schema;
- exact 40-character pinned source revision;
- catalog/source bundle count agreement;
- safe bundle IDs, aliases, filenames and upstream paths;
- alias uniqueness;
- exactly one static `bundle_metadata`, `tokenizer`, and `bos_conditioning` artifact;
- at least one quality for every required ONNX component;
- no duplicate `(role, quality)` pair;
- every profile references an actually present `(role, quality)` pair;
- URLs match repository/revision/path exactly;
- positive size and valid SHA-256 when the builder can obtain them;
- deterministic output for the same upstream revision.

CLI:

```text
onnxvoice catalog pocket build --output ... --source-output ...
onnxvoice catalog pocket verify --catalog ... --source ...
```

Also remove hard-coded system choices from normal `list`/`import` CLI paths where practical;
derive choices from the registered systems/catalog systems.

## Tests

Add at least:

```text
tests/test_pocket_catalog.py
tests/test_pocket_catalog_tools.py
tests/test_pocket_adapter.py
tests/test_pocket_local_runtime.py
tests/test_pocket_network_boundaries.py
```

Coverage targets:

- catalog normalizes semantic roles and profiles;
- `quality="int8"` selects INT8 Flow-LM/decoder plus FP32 encoder/conditioner;
- installs preserve role/quality/format metadata;
- `open()` is strictly local after installation;
- `open_local(files=...)` performs no catalog/network access;
- Mimi encoder is lazy;
- all created sessions close;
- fake sessions validate state-manifest wiring and output naming;
- `prepare_voice()` state can be reused for multiple `infer()` calls;
- token limit violations produce an explicit runtime contract error if Pocketsynth fails to split;
- provider options are propagated consistently to all component sessions.

## Acceptance API

After implementation, this should work:

```python
from onnxvoice import OnnxVoice

ov = OnnxVoice()
installation = ov.install("pocket:english_2026-04", quality="int8")

with ov.open(installation, providers="CPUExecutionProvider") as runtime:
    voice = runtime.prepare_voice(reference_audio, sample_rate=24000)
    result = runtime.infer(token_ids, voice_state=voice, temperature=0.7, lsd_steps=1)
    assert result.sample_rate == 24000
```

And this must remain network-free:

```python
with onnxvoice.open_local(
    system="pocket",
    files={
        "bundle_metadata": "bundle.json",
        "bos_conditioning": "bos_before_voice.npy",
        "flow_lm_main": "flow_lm_main_int8.onnx",
        "flow_lm_flow": "flow_lm_flow_int8.onnx",
        "mimi_decoder": "mimi_decoder_int8.onnx",
        "mimi_encoder": "mimi_encoder.onnx",
        "text_conditioner": "text_conditioner.onnx",
    },
    sample_rate=24000,
) as runtime:
    ...
```

The tokenizer path is intentionally absent from the runtime file map because tokenization belongs
to Pocketsynth's Pocket frontend, not the OnnxVoice graph adapter. Managed installations still
store the tokenizer artifact so Pocketsynth can locate it from the installation.

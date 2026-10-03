---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: v0.2.2
kind: added
summary: Added typed metadata-only voice discovery and pinned managed prompt preparation
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0012
paths:
  - pocketsynth/voice_prompts.py
  - pocketsynth/discovery.py
  - pocketsynth/voice.py
  - pocketsynth/runtime.py
  - pocketsynth/__init__.py
  - README.md
  - docs/architecture.md
  - examples/kyutai_voice.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 3
---

Public prompt inspection and bundle discovery return PocketSynth-owned metadata without fetching WAV payloads, installing bundles, or opening a runtime. VoicePromptInfo pins the catalog SHA-256 and prompt revision; pinned preparation verifies the current identity before fetching and keeps prompt provenance distinct from bundle revision and the normalized prepared-audio fingerprint. OnnxVoice continues to own catalog access, cache/download integrity, and runtime primitives, with the supported minimum remaining 0.2.2.

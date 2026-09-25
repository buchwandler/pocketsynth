---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0005
release_version: v0.2.0
kind: added
summary:
  Added opt-in static voice-level calibration from bundle metadata or an explicit
  gain override
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - pocketsynth/voice_level.py
  - pocketsynth/runtime.py
  - pocketsynth/types.py
issues: []
prs: []
sources:
  - git:5ed004dd39e0c41d3b11f5a6a9ec8b6c83cc52c4
contributors:
  - "@holgern"
breaking: false
internal: false
order: 5
---

VoiceLevelConfig defaults to no gain adjustment and records the applied source in result metadata. PocketSynth does not measure or normalize loudness per request.

---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.2.0
kind: changed
summary:
  Changed the PocketSynth engine boundary to use PocketRuntime for already-prepared
  speakable text
status: accepted
audience: null
scopes: []
source_refs:
  - git:9eabc2ea46c46580a988757af52fcd4c51eefc4a
paths:
  - pocketsynth/runtime.py
  - pocketsynth/convenience.py
  - README.md
  - docs/architecture.md
issues: []
prs: []
sources: []
contributors:
  - "@holgern"
breaking: true
internal: false
order: 2
---

PocketRuntime is the direct engine boundary for one synthesis request. Document planning, semantic preparation, logical voice roles, pauses, markers, and composition are caller-owned.

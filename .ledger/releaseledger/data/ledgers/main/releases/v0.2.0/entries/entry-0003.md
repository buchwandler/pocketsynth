---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: v0.2.0
kind: removed
summary:
  Removed planner, document-streaming, composition, and compatibility APIs,
  along with their runtime package dependencies
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - pocketsynth/__init__.py
  - pocketsynth/pipeline.py
  - pocketsynth/config.py
  - pocketsynth/types.py
  - pyproject.toml
issues: []
prs: []
sources:
  - git:9eabc2ea46c46580a988757af52fcd4c51eefc4a
contributors:
  - "@holgern"
breaking: true
internal: false
order: 3
---

UtterPlan, AudioCompose, and SSMD are no longer required by PocketSynth. OnnxVoice CPU and GPU extras now match the supported Pocket API range.

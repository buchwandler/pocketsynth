---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0004
release_version: v0.2.0
kind: changed
summary:
  Changed PocketRuntime.synthesize() to a strict single-request API with one
  encoding and at most one inference
status: accepted
audience: null
scopes: []
source_refs:
  - git:5ed004dd39e0c41d3b11f5a6a9ec8b6c83cc52c4
paths:
  - pocketsynth/runtime.py
  - pocketsynth/types.py
  - pocketsynth/errors.py
  - pocketsynth/__init__.py
  - README.md
issues: []
prs: []
sources: []
contributors:
  - "@holgern"
breaking: true
internal: false
order: 4
---

SynthesisRequest and SynthesisResult define the strict boundary. Model-limit chunking remains an explicit operation outside this atomic method. The API now reports typed request errors and bundle-language incompatibility.

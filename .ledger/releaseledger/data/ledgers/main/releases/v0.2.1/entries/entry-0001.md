---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.1
kind: changed
summary: Changed the supported OnnxVoice range to include 0.2.x releases
status: accepted
audience: null
scopes: []
source_refs:
  - git:210d280ae5c8464038069243595e1ce6e1233e07
paths:
  - .github/workflows/python-publish.yml
  - .github/workflows/tests.yml
  - docs/onnxvoice-pocket-support-implementation-brief.md
  - pyproject.toml
  - tests/test_package_boundary.py
issues: []
prs: []
sources:
  - git:210d280ae5c8464038069243595e1ce6e1233e07
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

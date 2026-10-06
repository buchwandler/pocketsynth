---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.2.5
kind: changed
summary: Changed the minimum supported OnnxVoice version to 0.2.4
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - .github/workflows/python-publish.yml
  - .github/workflows/tests.yml
  - docs/architecture.md
  - pocketsynth/_onnxvoice.py
  - pyproject.toml
  - tests/test_package_boundary.py
issues: []
prs: []
sources:
  - git:41cfd3cb4586cf738820fac082efbe89312e8345
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---

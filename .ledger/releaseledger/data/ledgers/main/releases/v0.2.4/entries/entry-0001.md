---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.4
kind: added
summary:
  Added a versioned public request API contract, request measurement, and validated
  discovery metadata
status: accepted
audience: null
scopes: []
source_refs:
  - git:c7d5f897fa09ee141c79d1aa207a605496c343e9
paths:
  - .github/workflows/tests.yml
  - README.md
  - docs/architecture.md
  - pocketsynth/__init__.py
  - pocketsynth/api_contract.py
  - pocketsynth/discovery.py
  - pocketsynth/runtime.py
  - pocketsynth/types.py
  - tests/test_discovery.py
  - tests/test_request_api_contract.py
  - tests/test_runtime_strict.py
  - tests/test_types.py
  - tests/wheel_api_smoke.py
issues: []
prs: []
sources:
  - git:c7d5f897fa09ee141c79d1aa207a605496c343e9
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.2
kind: added
summary:
  Added managed Kyutai voice prompt discovery and synthesis through the runtime
  and CLI
status: accepted
audience: null
scopes: []
source_refs:
  - git:23bcc4ef7cb0d9b9f11fd160b433da51e1dd221f
paths:
  - .github/workflows/python-publish.yml
  - .github/workflows/tests.yml
  - README.md
  - examples/README.md
  - examples/clone_all_kyutai_voices.py
  - examples/kyutai_voice.py
  - examples/run_all.py
  - pocketsynth/__main__.py
  - pocketsynth/_onnxvoice.py
  - pocketsynth/runtime.py
  - pocketsynth/voice.py
  - pyproject.toml
  - tests/integration/test_real_managed_wav.py
  - tests/test_cli.py
  - tests/test_examples.py
  - tests/test_onnxvoice_boundary.py
  - tests/test_package_boundary.py
  - tests/test_runtime_synthesis.py
  - tests/test_voice.py
issues: []
prs: []
sources:
  - git:23bcc4ef7cb0d9b9f11fd160b433da51e1dd221f
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

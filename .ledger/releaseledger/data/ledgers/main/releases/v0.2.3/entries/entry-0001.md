---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.3
kind: added
summary:
  Added metadata-only bundle and voice prompt discovery with pinned managed
  voice preparation
status: accepted
audience: null
scopes: []
source_refs:
  - git:d030901884294fd61e8639e4a0c0cdea513cc748
paths:
  - .ledger/releaseledger/data/ledgers/main/events/events.jsonl
  - .ledger/releaseledger/data/ledgers/main/releases/v0.2.2/entries/entry-0003.md
  - .ledger/releaseledger/data/ledgers/main/releases/v0.2.2/release.md
  - README.md
  - docs/architecture.md
  - docs/changelog.md
  - examples/README.md
  - examples/clone_all_kyutai_voices.py
  - examples/kyutai_voice.py
  - pocketsynth/__init__.py
  - pocketsynth/__main__.py
  - pocketsynth/_onnxvoice.py
  - pocketsynth/discovery.py
  - pocketsynth/errors.py
  - pocketsynth/runtime.py
  - pocketsynth/voice.py
  - pocketsynth/voice_prompts.py
  - tests/test_cli.py
  - tests/test_discovery.py
  - tests/test_examples.py
  - tests/test_onnxvoice_boundary.py
  - tests/test_package_boundary.py
  - tests/test_runtime_strict.py
  - tests/test_voice.py
  - tests/test_voice_prompts.py
issues: []
prs: []
sources:
  - git:d030901884294fd61e8639e4a0c0cdea513cc748
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

"""Synthesize with the bundle-declared Pocket voice alba."""

from __future__ import annotations

import os

from _output import artefact_path

from pocketsynth import PocketRuntime

BUNDLE = os.environ.get("POCKETSYNTH_EXAMPLE_BUNDLE", "english_2026-04")
OFFLINE = os.environ.get("POCKETSYNTH_EXAMPLE_OFFLINE") == "1"


def main() -> int:
    output = artefact_path("predefined_voice_alba.wav")
    with PocketRuntime.from_pretrained(BUNDLE, offline=OFFLINE) as runtime:
        if "alba" not in runtime.predefined_voices:
            available = ", ".join(runtime.predefined_voices) or "none"
            raise SystemExit(
                f"Bundle {BUNDLE!r} does not declare alba. Available voices: {available}."
            )
        result = runtime.synthesize_text("Hello from Pocket.", voice="alba")
        result.save_wav(output)

    print(f"Bundle: {BUNDLE}")
    print("Voice: alba")
    print(f"WAV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

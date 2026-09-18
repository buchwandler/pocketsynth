from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .errors import BundleNotFoundError, UnsupportedBundleError

Precision = Literal["int8", "fp32"]


@dataclass(frozen=True, slots=True)
class BundleMetadata:
    path: Path
    bundle_name: str
    language: str
    schema_version: int
    sample_rate: int
    samples_per_frame: int
    max_token_per_chunk: int
    tokenizer_file: str
    bos_before_voice_file: str
    remove_semicolons: bool = False
    pad_with_spaces_for_short_inputs: bool = False
    model_recommended_frames_after_eos: int | None = None
    predefined_voices: tuple[str, ...] = ()
    raw: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "BundleMetadata":
        source = Path(path)
        if not source.is_file():
            raise BundleNotFoundError(f"Missing Pocket bundle metadata: {source}")
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UnsupportedBundleError(f"Invalid Pocket bundle metadata: {source}") from exc
        required = {
            "bundle_name",
            "language",
            "schema_version",
            "sample_rate",
            "samples_per_frame",
            "max_token_per_chunk",
            "tokenizer_file",
            "bos_before_voice_file",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise UnsupportedBundleError(
                f"Pocket bundle metadata is missing: {', '.join(missing)}"
            )
        return cls(
            path=source.resolve(),
            bundle_name=str(raw["bundle_name"]),
            language=str(raw["language"]),
            schema_version=int(raw["schema_version"]),
            sample_rate=int(raw["sample_rate"]),
            samples_per_frame=int(raw["samples_per_frame"]),
            max_token_per_chunk=int(raw["max_token_per_chunk"]),
            tokenizer_file=str(raw["tokenizer_file"]),
            bos_before_voice_file=str(raw["bos_before_voice_file"]),
            remove_semicolons=bool(raw.get("remove_semicolons", False)),
            pad_with_spaces_for_short_inputs=bool(raw.get("pad_with_spaces_for_short_inputs", False)),
            model_recommended_frames_after_eos=(
                int(raw["model_recommended_frames_after_eos"])
                if raw.get("model_recommended_frames_after_eos") is not None
                else None
            ),
            predefined_voices=tuple(str(v) for v in raw.get("predefined_voices", ())),
            raw=dict(raw),
        )

    @property
    def frame_duration(self) -> float:
        return self.samples_per_frame / self.sample_rate


@dataclass(frozen=True, slots=True)
class BundlePaths:
    root: Path
    metadata: BundleMetadata
    tokenizer: Path
    bos_conditioning: Path
    flow_lm_main: Path
    flow_lm_flow: Path
    mimi_decoder: Path
    mimi_encoder: Path
    text_conditioner: Path
    precision: Precision

    @classmethod
    def from_directory(
        cls, directory: str | Path, *, precision: Precision = "int8"
    ) -> "BundlePaths":
        root = Path(directory).expanduser().resolve()
        metadata = BundleMetadata.load(root / "bundle.json")
        if precision not in {"int8", "fp32"}:
            raise ValueError("precision must be 'int8' or 'fp32'")

        def required(name: str) -> Path:
            path = root / name
            if not path.is_file():
                raise BundleNotFoundError(f"Missing Pocket bundle artifact: {path}")
            return path

        suffix = "_int8.onnx" if precision == "int8" else ".onnx"
        # Match the current upstream Python runtime profile: encoder and text conditioner stay FP32.
        return cls(
            root=root,
            metadata=metadata,
            tokenizer=required(metadata.tokenizer_file),
            bos_conditioning=required(metadata.bos_before_voice_file),
            flow_lm_main=required(f"flow_lm_main{suffix}"),
            flow_lm_flow=required(f"flow_lm_flow{suffix}"),
            mimi_decoder=required(f"mimi_decoder{suffix}"),
            mimi_encoder=required("mimi_encoder.onnx"),
            text_conditioner=required("text_conditioner.onnx"),
            precision=precision,
        )

    def runtime_files(self) -> dict[str, Path]:
        return {
            "bundle_metadata": self.metadata.path,
            "bos_conditioning": self.bos_conditioning,
            "flow_lm_main": self.flow_lm_main,
            "flow_lm_flow": self.flow_lm_flow,
            "mimi_decoder": self.mimi_decoder,
            "mimi_encoder": self.mimi_encoder,
            "text_conditioner": self.text_conditioner,
        }

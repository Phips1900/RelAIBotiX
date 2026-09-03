"""Small registry for pretrained skill-detector checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.resources
import json
import os
from pathlib import Path
from typing import Mapping

from relaibotix.data import inspect_h5


@dataclass(frozen=True)
class DetectorSpec:
    detector_id: str
    case_study: str
    modality: str
    checkpoint: str
    required_features: tuple[str, ...]
    recommended: bool = False


@dataclass(frozen=True)
class DetectorRegistry:
    detectors: Mapping[str, DetectorSpec]


def load_registry(path: str | Path | None = None) -> DetectorRegistry:
    """Load the bundled registry or an explicitly supplied replacement."""

    if path is None:
        text = (
            importlib.resources.files("relaibotix.skilldetector")
            .joinpath("checkpoints.json")
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text)
    if raw.get("schema_version") != "1.0":
        raise ValueError("Checkpoint registry schema_version must be '1.0'.")
    schemas = raw.get("feature_schemas")
    definitions = raw.get("detectors")
    if not isinstance(schemas, dict) or not isinstance(definitions, dict):
        raise ValueError("Checkpoint registry requires feature_schemas and detectors objects.")

    detectors: dict[str, DetectorSpec] = {}
    for detector_id, definition in definitions.items():
        if not isinstance(definition, dict):
            raise ValueError(f"Detector '{detector_id}' must be an object.")
        missing = {"case_study", "modality", "checkpoint", "feature_schema"} - set(definition)
        if missing:
            raise ValueError(
                f"Detector '{detector_id}' is missing fields: {', '.join(sorted(missing))}."
            )
        modality = str(definition["modality"])
        if modality not in {"timeseries", "camera", "hybrid"}:
            raise ValueError(f"Detector '{detector_id}' has unsupported modality '{modality}'.")
        schema_name = str(definition["feature_schema"])
        features = schemas.get(schema_name)
        if not isinstance(features, list) or not features or not all(
            isinstance(name, str) and name for name in features
        ):
            raise ValueError(
                f"Detector '{detector_id}' references invalid feature schema '{schema_name}'."
            )
        detectors[str(detector_id)] = DetectorSpec(
            detector_id=str(detector_id),
            case_study=str(definition["case_study"]),
            modality=modality,
            checkpoint=str(definition["checkpoint"]),
            required_features=tuple(features),
            recommended=bool(definition.get("recommended", False)),
        )
    if not detectors:
        raise ValueError("Checkpoint registry contains no detectors.")
    return DetectorRegistry(detectors)


def select_detector(
    registry: DetectorRegistry,
    h5_path: str | Path,
    *,
    detector_id: str | None = None,
    case_study: str | None = None,
    modality: str | None = None,
) -> DetectorSpec:
    """Select one compatible detector and reject HDF5 schema mismatches early."""

    available_features = set(inspect_h5(h5_path).feature_names)
    if detector_id is not None:
        if detector_id not in registry.detectors:
            raise ValueError(
                f"Unknown detector '{detector_id}'. Available: "
                + ", ".join(sorted(registry.detectors))
            )
        candidates = [registry.detectors[detector_id]]
    else:
        # HDF5-only automatic inference defaults to time series. Camera and
        # hybrid inference remain explicit because they need an aligned video root.
        selected_modality = modality or "timeseries"
        candidates = [
            detector
            for detector in registry.detectors.values()
            if detector.modality == selected_modality
            and (case_study is None or detector.case_study == case_study)
            and set(detector.required_features).issubset(available_features)
        ]
        recommended = [detector for detector in candidates if detector.recommended]
        if len(recommended) == 1:
            return recommended[0]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                "More than one compatible detector was found; select one with --detector."
            )
        raise ValueError(
            "No compatible pretrained detector was found for this HDF5 feature schema."
        )

    detector = candidates[0]
    if case_study is not None and detector.case_study != case_study:
        raise ValueError(
            f"Detector '{detector.detector_id}' belongs to '{detector.case_study}', not '{case_study}'."
        )
    if modality is not None and detector.modality != modality:
        raise ValueError(
            f"Detector '{detector.detector_id}' uses modality '{detector.modality}', not '{modality}'."
        )
    missing = sorted(set(detector.required_features) - available_features)
    if missing:
        raise ValueError(
            f"Detector '{detector.detector_id}' is incompatible with the HDF5 input; "
            f"missing {len(missing)} features: {', '.join(missing)}."
        )
    return detector


def resolve_checkpoint(
    detector: DetectorSpec,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Resolve a registered checkpoint without downloading or copying models."""

    root_value = checkpoint_root or os.environ.get("RELAIBOTIX_CHECKPOINT_ROOT")
    root = Path(root_value) if root_value is not None else Path("artifacts/checkpoints")
    checkpoint = root / detector.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint for '{detector.detector_id}' was not found at {checkpoint}. "
            "Provide --checkpoint-root or set RELAIBOTIX_CHECKPOINT_ROOT."
        )
    return checkpoint

"""Skill-detector inference API.

Model training is maintained in its separate training repository.
"""

from .inference import InferenceResult, run_inference
from .registry import (
    DetectorRegistry,
    DetectorSpec,
    load_registry,
    resolve_checkpoint,
    select_detector,
)

__all__ = [
    "DetectorRegistry",
    "DetectorSpec",
    "InferenceResult",
    "load_registry",
    "resolve_checkpoint",
    "run_inference",
    "select_detector",
]

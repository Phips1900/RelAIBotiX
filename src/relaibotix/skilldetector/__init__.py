"""Skill-detector inference API.

Model training is maintained in its separate training repository.
"""

from .inference import CANONICAL_SKILL_DATASET, InferenceResult, run_inference

__all__ = ["CANONICAL_SKILL_DATASET", "InferenceResult", "run_inference"]

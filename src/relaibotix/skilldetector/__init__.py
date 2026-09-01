"""Skill-detector inference API.

Model training is maintained in its separate training repository.
"""

from .inference import InferenceResult, run_inference

__all__ = ["InferenceResult", "run_inference"]

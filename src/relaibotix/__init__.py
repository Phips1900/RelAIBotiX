from importlib.metadata import version, PackageNotFoundError

from .behavioral import BehavioralAnalyzer, BehavioralResult, BehavioralThresholds

try:
    from .reliability.reliability_models import HybridReliabilityModel, MarkovChain, FaultTree  # noqa: F401
except Exception:
    HybridReliabilityModel = MarkovChain = FaultTree = None  # type: ignore

__all__ = [
    "BehavioralAnalyzer",
    "BehavioralResult",
    "BehavioralThresholds",
    "HybridReliabilityModel",
    "MarkovChain",
    "FaultTree",
]

try:
    __version__ = version("relaibotix")
except PackageNotFoundError:
    __version__ = "0.0.0"

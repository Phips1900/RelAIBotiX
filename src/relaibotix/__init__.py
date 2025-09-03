from importlib.metadata import version, PackageNotFoundError

# Public API shortcuts
try:
    from .behavioral.behavioral_analysis import BehavioralAnalyzer  # noqa: F401
except Exception:
    BehavioralAnalyzer = None  # type: ignore

try:
    from .reliability.reliability_models import HybridReliabilityModel, MarkovChain, FaultTree  # noqa: F401
except Exception:
    HybridReliabilityModel = MarkovChain = FaultTree = None  # type: ignore

__all__ = [
    "BehavioralAnalyzer",
    "HybridReliabilityModel",
    "MarkovChain",
    "FaultTree",
]

try:
    __version__ = version("relaibotix")
except PackageNotFoundError:
    __version__ = "0.0.0"

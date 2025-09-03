try:
    from .behavioral_analysis import BehavioralAnalyzer  # noqa: F401
except Exception:
    BehavioralAnalyzer = None  # type: ignore

__all__ = ["BehavioralAnalyzer"]
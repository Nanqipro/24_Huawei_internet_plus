"""GOODLAB driver fatigue and distraction detection toolkit.

The lightweight data and event APIs remain importable without OpenCV or ONNX
Runtime. Inference dependencies are loaded only when inference is requested.
"""

from __future__ import annotations

from typing import Any

from .events import BehaviorEvent, TemporalEventTracker, classify_frame
from .types import CLASS_NAMES, Detection

__all__ = [
    "CLASS_NAMES",
    "BehaviorEvent",
    "Detection",
    "OnnxDetector",
    "TemporalEventTracker",
    "analyze_video",
    "classify_frame",
]

__version__ = "0.1.0"


def analyze_video(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily import and run the full video pipeline."""
    from .pipeline import analyze_video as _analyze_video

    return _analyze_video(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "OnnxDetector":
        from .detector import OnnxDetector

        return OnnxDetector
    raise AttributeError(name)

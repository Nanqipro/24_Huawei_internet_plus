"""Dependency-light public data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CLASS_NAMES = (
    "face",
    "phone",
    "open_eye",
    "closed_eye",
    "yawn",
    "closed_mouth",
    "turned_head",
    "lowered_head",
)


@dataclass(frozen=True)
class Detection:
    """A driver-related detection in center-based image coordinates."""

    class_id: int
    label: str
    box_xywh: tuple[float, float, float, float]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "box_xywh": [round(value, 3) for value in self.box_xywh],
            "confidence": round(self.confidence, 6),
        }


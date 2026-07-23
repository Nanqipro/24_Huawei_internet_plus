"""Frame-level behavior classification and temporal event aggregation."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Detection

BEHAVIOR_CATEGORY = {
    "closed_eyes": 1,
    "yawning": 2,
    "using_phone": 3,
    "looking_away": 4,
}


@dataclass(frozen=True)
class BehaviorEvent:
    """A behavior that stayed active long enough to be reported."""

    category: int
    label: str
    start_ms: int
    end_ms: int

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "label": self.label,
            "period_ms": [self.start_ms, self.end_ms],
        }


def classify_frame(detections: list[Detection]) -> set[str]:
    """Map model classes in one frame to competition behavior labels."""
    class_ids = {item.class_id for item in detections}
    behaviors: set[str] = set()

    if 3 in class_ids:
        behaviors.add("closed_eyes")
    if 4 in class_ids and 5 not in class_ids:
        behaviors.add("yawning")
    if 1 in class_ids:
        behaviors.add("using_phone")

    face_visible = bool(class_ids.intersection({0, 6, 7}))
    head_turned = bool(class_ids.intersection({6, 7}))
    if (head_turned or not face_visible) and "using_phone" not in behaviors:
        behaviors.add("looking_away")

    return behaviors


class TemporalEventTracker:
    """Aggregate sampled frame states into events with a minimum duration."""

    def __init__(self, *, min_duration_ms: int = 3000, sample_interval_ms: int = 333):
        if min_duration_ms <= 0:
            raise ValueError("min_duration_ms must be positive")
        if sample_interval_ms <= 0:
            raise ValueError("sample_interval_ms must be positive")
        self.min_duration_ms = min_duration_ms
        self.sample_interval_ms = sample_interval_ms
        self._active_since: dict[str, int] = {}
        self._last_seen: dict[str, int] = {}
        self._events: list[BehaviorEvent] = []

    def update(self, timestamp_ms: int, active_behaviors: set[str]) -> None:
        """Update the tracker with the behaviors active at one sample."""
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms cannot be negative")
        unknown = active_behaviors.difference(BEHAVIOR_CATEGORY)
        if unknown:
            raise ValueError(f"unknown behaviors: {sorted(unknown)}")

        for label in BEHAVIOR_CATEGORY:
            if label in active_behaviors:
                self._active_since.setdefault(label, timestamp_ms)
                self._last_seen[label] = timestamp_ms
            elif label in self._active_since:
                self._close(label)

    def finish(self) -> list[BehaviorEvent]:
        """Close all open behaviors and return events in timeline order."""
        for label in list(self._active_since):
            self._close(label)
        return sorted(self._events, key=lambda event: (event.start_ms, event.category))

    def _close(self, label: str) -> None:
        start_ms = self._active_since.pop(label)
        last_seen_ms = self._last_seen.pop(label)
        end_ms = last_seen_ms + self.sample_interval_ms
        if end_ms - start_ms >= self.min_duration_ms:
            self._events.append(
                BehaviorEvent(
                    category=BEHAVIOR_CATEGORY[label],
                    label=label,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )

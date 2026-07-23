import unittest

from goodlab_fatigue_detection.events import TemporalEventTracker, classify_frame
from goodlab_fatigue_detection.types import Detection


def detection(class_id: int) -> Detection:
    labels = {
        0: "face",
        1: "phone",
        3: "closed_eye",
        4: "yawn",
        5: "closed_mouth",
        6: "turned_head",
    }
    return Detection(class_id, labels[class_id], (50.0, 50.0, 20.0, 20.0), 0.9)


class ClassifyFrameTests(unittest.TestCase):
    def test_maps_closed_eyes_and_yawning(self) -> None:
        behaviors = classify_frame([detection(0), detection(3), detection(4)])
        self.assertEqual(behaviors, {"closed_eyes", "yawning"})

    def test_closed_mouth_suppresses_yawn(self) -> None:
        behaviors = classify_frame([detection(0), detection(4), detection(5)])
        self.assertNotIn("yawning", behaviors)

    def test_phone_takes_precedence_over_looking_away(self) -> None:
        behaviors = classify_frame([detection(1), detection(6)])
        self.assertIn("using_phone", behaviors)
        self.assertNotIn("looking_away", behaviors)


class TemporalEventTrackerTests(unittest.TestCase):
    def test_keeps_event_at_minimum_duration(self) -> None:
        tracker = TemporalEventTracker(min_duration_ms=3000, sample_interval_ms=1000)
        tracker.update(0, {"closed_eyes"})
        tracker.update(1000, {"closed_eyes"})
        tracker.update(2000, {"closed_eyes"})
        tracker.update(3000, set())

        self.assertEqual(
            [event.as_dict() for event in tracker.finish()],
            [
                {
                    "category": 1,
                    "label": "closed_eyes",
                    "period_ms": [0, 3000],
                }
            ],
        )

    def test_drops_short_event(self) -> None:
        tracker = TemporalEventTracker(min_duration_ms=3000, sample_interval_ms=1000)
        tracker.update(0, {"yawning"})
        tracker.update(1000, {"yawning"})
        tracker.update(2000, set())
        self.assertEqual(tracker.finish(), [])


if __name__ == "__main__":
    unittest.main()

import unittest

from goodlab_fatigue_detection.postprocess import (
    filter_driver_detections,
    intersection_over_union,
    xyxy_to_xywh,
)


class GeometryTests(unittest.TestCase):
    def test_xyxy_to_xywh(self) -> None:
        self.assertEqual(xyxy_to_xywh([10, 20, 30, 50]), [20.0, 35.0, 20.0, 30.0])

    def test_iou_for_overlapping_boxes(self) -> None:
        self.assertAlmostEqual(
            intersection_over_union([0, 0, 10, 10], [5, 5, 15, 15]),
            25 / 175,
        )

    def test_iou_handles_empty_boxes(self) -> None:
        self.assertEqual(intersection_over_union([0, 0, 0, 0], [0, 0, 0, 0]), 0)


class DriverFilterTests(unittest.TestCase):
    def test_keeps_face_eye_and_phone_near_driver(self) -> None:
        rows = [
            (0, [150.0, 80.0, 100.0, 100.0], 0.95),
            (3, [140.0, 70.0, 20.0, 10.0], 0.90),
            (1, [150.0, 175.0, 30.0, 30.0], 0.80),
            (3, [20.0, 20.0, 10.0, 10.0], 0.99),
        ]
        result = filter_driver_detections(
            rows,
            frame_width=300,
            frame_height=200,
        )
        self.assertEqual([row[0] for row in result], [0, 3, 1])

    def test_returns_empty_without_driver_face(self) -> None:
        rows = [(1, [150.0, 175.0, 30.0, 30.0], 0.80)]
        self.assertEqual(
            filter_driver_detections(rows, frame_width=300, frame_height=200),
            [],
        )


if __name__ == "__main__":
    unittest.main()


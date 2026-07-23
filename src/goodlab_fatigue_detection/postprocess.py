"""Geometry and driver-focused detection filtering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

DetectionRow = tuple[int, list[float], float]


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    """Convert ``[x1, y1, x2, y2]`` to center-based ``[x, y, w, h]``."""
    if len(box) != 4:
        raise ValueError("box must contain exactly four coordinates")
    x1, y1, x2, y2 = (float(value) for value in box)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    """Convert center-based ``[x, y, w, h]`` to ``[x1, y1, x2, y2]``."""
    if len(box) != 4:
        raise ValueError("box must contain exactly four coordinates")
    x, y, width, height = (float(value) for value in box)
    return [
        x - width / 2,
        y - height / 2,
        x + width / 2,
        y + height / 2,
    ]


def intersection_over_union(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return the IoU of two ``xyxy`` boxes, safely handling empty boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("each box must contain exactly four coordinates")

    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)

    intersection_width = max(min(ax2, bx2) - max(ax1, bx1), 0.0)
    intersection_height = max(min(ay2, by2) - max(ay1, by1), 0.0)
    intersection = intersection_width * intersection_height
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def filter_driver_detections(
    detections: Iterable[DetectionRow],
    *,
    frame_width: float,
    frame_height: float,
    min_face_area_ratio: float = 0.05,
) -> list[DetectionRow]:
    """Keep detections associated with the most prominent driver face.

    The competition model uses class ``0`` for a frontal face and may emit
    classes ``6``/``7`` for turned or lowered heads. Eyes and mouth detections
    must overlap the selected face; a phone may overlap either the face or the
    area immediately below it.
    """
    rows = list(detections)
    if not rows or frame_width <= 0 or frame_height <= 0:
        return []

    frame_area = frame_width * frame_height
    face_class_ids = {0, 6, 7}
    driver_index = -1
    largest_face_area = 0.0

    for index, (class_id, box, _confidence) in enumerate(rows):
        if class_id not in face_class_ids:
            continue
        area = max(float(box[2]), 0.0) * max(float(box[3]), 0.0)
        if area > largest_face_area and area / frame_area > min_face_area_ratio:
            driver_index = index
            largest_face_area = area

    if driver_index < 0:
        return []

    _driver_class, driver_box, _driver_confidence = rows[driver_index]
    center_x, center_y, face_width, face_height = driver_box
    face_box = [
        max(center_x - face_width / 2, 0.0),
        max(center_y - face_height / 2, 0.0),
        min(center_x + face_width / 2, frame_width),
        min(center_y + face_height / 2, frame_height),
    ]
    phone_area = [
        max(center_x - face_width, 0.0),
        (frame_height + face_box[3]) / 2,
        min(center_x + face_width, frame_width),
        frame_height,
    ]

    clean: list[DetectionRow] = []
    for row in rows:
        class_id, box, _confidence = row
        object_box = xywh_to_xyxy(box)
        face_iou = intersection_over_union(object_box, face_box)

        if class_id in face_class_ids and face_iou > 0.5:
            clean.append(row)
        elif class_id in {2, 3, 4, 5} and face_iou > 0:
            clean.append(row)
        elif class_id == 1 and (
            face_iou > 0 or intersection_over_union(object_box, phone_area) > 0
        ):
            clean.append(row)

    return clean


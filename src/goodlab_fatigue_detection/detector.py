"""ONNX Runtime detector used by the GOODLAB competition pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .postprocess import filter_driver_detections, xyxy_to_xywh
from .types import CLASS_NAMES, Detection


class OnnxDetector:
    """Load the competition model and detect driver-related visual cues."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int = 320,
        use_cuda: bool = False,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "onnxruntime is required for inference; install the project first"
            ) from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        if input_size <= 0:
            raise ValueError("input_size must be positive")

        available = set(ort.get_available_providers())
        requested = ["CPUExecutionProvider"]
        if use_cuda and "CUDAExecutionProvider" in available:
            requested.insert(0, "CUDAExecutionProvider")

        self.model_path = path
        self.input_size = input_size
        self.session = ort.InferenceSession(str(path), providers=requested)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one BGR frame and return driver-focused detections."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty BGR image with three channels")

        tensor, ratio, padding = self._letterbox(frame)
        output = self.session.run(self.output_names, {self.input_name: tensor})[0]
        raw_rows: list[tuple[int, list[float], float]] = []

        for row in np.asarray(output).reshape(-1, 7):
            _batch_id, x1, y1, x2, y2, class_id, confidence = row.tolist()
            box = np.asarray([x1, y1, x2, y2], dtype=np.float32)
            box -= np.asarray(padding * 2, dtype=np.float32)
            box /= ratio
            xywh = xyxy_to_xywh(box.round().tolist())
            raw_rows.append((int(class_id), xywh, float(confidence)))

        frame_height, frame_width = frame.shape[:2]
        clean_rows = filter_driver_detections(
            raw_rows,
            frame_width=float(frame_width),
            frame_height=float(frame_height),
        )
        return [
            Detection(
                class_id=class_id,
                label=(
                    CLASS_NAMES[class_id]
                    if 0 <= class_id < len(CLASS_NAMES)
                    else f"class_{class_id}"
                ),
                box_xywh=tuple(float(value) for value in box),
                confidence=confidence,
            )
            for class_id, box, confidence in clean_rows
        ]

    def _letterbox(
        self,
        frame: np.ndarray,
        *,
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        ratio = min(self.input_size / height, self.input_size / width)
        resized_width = int(round(width * ratio))
        resized_height = int(round(height * ratio))
        pad_width = self.input_size - resized_width
        pad_height = self.input_size - resized_height
        half_width = pad_width / 2
        half_height = pad_height / 2

        if (width, height) != (resized_width, resized_height):
            image = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )

        top = int(round(half_height - 0.1))
        bottom = int(round(half_height + 0.1))
        left = int(round(half_width - 0.1))
        right = int(round(half_width + 0.1))
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )
        tensor = image.transpose((2, 0, 1))[None, ...]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
        return tensor, ratio, (half_width, half_height)

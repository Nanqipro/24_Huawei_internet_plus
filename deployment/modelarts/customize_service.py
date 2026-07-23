"""Huawei Cloud ModelArts custom-service adapter.

This adapter expects the project package to be installed in the serving image
and the ONNX file to be placed beside the deployment bundle.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from goodlab_fatigue_detection import analyze_video
from model_service.pytorch_model_service import PTServingBaseService


class FatigueDrivingDetectionService(PTServingBaseService):
    """Accept a video upload and return temporally aggregated behavior events."""

    def __init__(self, model_name: str, model_path: str) -> None:
        super().__init__(model_name, model_path)
        model_location = Path(model_path)
        model_directory = (
            model_location if model_location.is_dir() else model_location.parent
        )
        self.model_file = model_directory / "fatigue-detection-v4-c7-320.onnx"
        self.input_video: Path | None = None

    def _preprocess(self, data: dict[str, Any]) -> dict[str, str]:
        for field in data.values():
            for filename, file_object in field.items():
                suffix = Path(filename).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(
                    prefix="goodlab-",
                    suffix=suffix,
                    delete=False,
                ) as temporary:
                    temporary.write(file_object.read())
                    self.input_video = Path(temporary.name)
                return {"status": "ready"}
        raise ValueError("request does not contain a video file")

    def _inference(self, _data: dict[str, str]) -> dict[str, Any]:
        if self.input_video is None:
            raise RuntimeError("preprocess must run before inference")
        try:
            return analyze_video(self.input_video, self.model_file)
        finally:
            self._remove_input_video()

    def _postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        self._remove_input_video()
        return data

    def _remove_input_video(self) -> None:
        if self.input_video is not None:
            try:
                os.unlink(self.input_video)
            except FileNotFoundError:
                pass
            self.input_video = None


# Preserve the class name used by the original competition bundle.
fatigue_driving_detection = FatigueDrivingDetectionService

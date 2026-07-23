"""Small OpenCV video reader with explicit metadata and cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_ms(self) -> int:
        if self.fps <= 0:
            return 0
        return int(round(self.frame_count / self.fps * 1000))


class VideoReader:
    """Context-managed reader that can advance by a fixed frame step."""

    def __init__(self, video_path: str | Path) -> None:
        self.path = Path(video_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")

        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"OpenCV could not open video: {self.path}")

        self.metadata = VideoMetadata(
            width=int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self.capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def read(self, step: int = 1) -> tuple[bool, np.ndarray | None]:
        if step <= 0:
            raise ValueError("step must be positive")
        for _ in range(step - 1):
            if not self.capture.grab():
                return False, None
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release()


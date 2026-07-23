"""End-to-end video analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from .detector import OnnxDetector
from .events import TemporalEventTracker, classify_frame
from .video import VideoReader


def analyze_video(
    video_path: str | Path,
    model_path: str | Path,
    *,
    samples_per_second: float = 3.0,
    min_event_seconds: float = 3.0,
    crop_start_ratio: float = 0.33,
    input_size: int = 320,
    use_cuda: bool = False,
    include_frames: bool = False,
) -> dict[str, Any]:
    """Analyze one driver video and return a JSON-serializable report."""
    if samples_per_second <= 0:
        raise ValueError("samples_per_second must be positive")
    if min_event_seconds <= 0:
        raise ValueError("min_event_seconds must be positive")
    if not 0 <= crop_start_ratio < 1:
        raise ValueError("crop_start_ratio must be in [0, 1)")

    detector = OnnxDetector(
        model_path,
        input_size=input_size,
        use_cuda=use_cuda,
    )
    started = perf_counter()
    frame_results: list[dict[str, Any]] = []

    with VideoReader(video_path) as reader:
        metadata = reader.metadata
        if metadata.fps <= 0:
            raise ValueError("video reports an invalid frame rate")

        frame_step = max(int(round(metadata.fps / samples_per_second)), 1)
        sample_interval_ms = max(int(round(frame_step / metadata.fps * 1000)), 1)
        tracker = TemporalEventTracker(
            min_duration_ms=int(round(min_event_seconds * 1000)),
            sample_interval_ms=sample_interval_ms,
        )

        frame_index = 0
        success, frame = reader.read()
        while success and frame is not None:
            timestamp_ms = int(round(frame_index / metadata.fps * 1000))
            crop_x = int(frame.shape[1] * crop_start_ratio)
            detections = detector.detect(frame[:, crop_x:, :])
            behaviors = classify_frame(detections)
            tracker.update(timestamp_ms, behaviors)

            if include_frames:
                frame_results.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "crop_offset_x": crop_x,
                        "behaviors": sorted(behaviors),
                        "detections": [item.as_dict() for item in detections],
                    }
                )
            frame_index += frame_step
            success, frame = reader.read(frame_step)

        events = tracker.finish()

    result: dict[str, Any] = {
        "video": Path(video_path).name,
        "video_duration_ms": metadata.duration_ms,
        "inference_duration_ms": int(round((perf_counter() - started) * 1000)),
        "sampling": {
            "requested_samples_per_second": samples_per_second,
            "frame_step": frame_step,
            "effective_interval_ms": sample_interval_ms,
        },
        "events": [event.as_dict() for event in events],
    }
    if include_frames:
        result["frames"] = frame_results
    return result

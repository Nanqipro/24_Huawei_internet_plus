"""Command-line interface for local video inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goodlab-fatigue",
        description="Analyze driver fatigue and distraction cues in a video.",
    )
    parser.add_argument("video", type=Path, help="input video path")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/fatigue-detection-v4-c7-320.onnx"),
        help="ONNX model path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this file instead of stdout",
    )
    parser.add_argument(
        "--samples-per-second",
        type=float,
        default=3.0,
        help="number of sampled frames per video second",
    )
    parser.add_argument(
        "--min-event-seconds",
        type=float,
        default=3.0,
        help="minimum continuous duration required for an event",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="use CUDAExecutionProvider when it is available",
    )
    parser.add_argument(
        "--include-frames",
        action="store_true",
        help="include per-frame detections in the JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .pipeline import analyze_video

    result = analyze_video(
        args.video,
        args.model,
        samples_per_second=args.samples_per_second,
        min_event_seconds=args.min_event_seconds,
        use_cuda=args.cuda,
        include_frames=args.include_frames,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved report to {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

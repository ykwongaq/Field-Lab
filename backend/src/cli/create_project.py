"""Create a VideoSegmenter project from a video or a folder of frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from src.core.config import DEFAULT_TARGET_FPS, get_settings
from src.core.storage import ensure_dir
from src.projects.builder import MODE_DESCRIPTIONS, MODES, create_project
from src.projects.layout import (
    DEFAULT_JPEG_QUALITY,
    PROJECT_EXTENSION,
    project_name_for,
)


def _read_metadata(
    raw: Optional[str], parser: argparse.ArgumentParser
) -> Dict[str, Any]:
    """Parse `--metadata`: either a JSON object or a path to a JSON file."""
    if not raw:
        return {}
    text = raw
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as handle:
            text = handle.read()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        parser.error(f"--metadata is not valid JSON: {exc}")
    if not isinstance(value, dict):
        parser.error("--metadata must be a JSON object (a dictionary)")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bundle a video or a folder of frames into a .project archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {mode}: {desc}" for mode, desc in MODE_DESCRIPTIONS.items()
        ),
    )
    parser.add_argument("video", nargs="?", help="Path to the source video file")
    parser.add_argument(
        "--frame_folder",
        help="Path to a folder of frames, instead of a video",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODES,
        help="Segmentation mode.",
    )
    parser.add_argument("--name", help="Project name")
    parser.add_argument(
        "--output_dir",
        help="Directory for the archive (default: the configured projects_dir)",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help="Frame rate the project should play at; 0 keeps the source rate",
    )
    parser.add_argument(
        "--original_fps",
        type=float,
        help="Frame rate of the frames, when it is not in the video metadata",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        help="Keep every N-th frame (overrides --target_fps)",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality 1-100 (video sources only)",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None, help="Cap on frames written"
    )
    parser.add_argument(
        "--annotation",
        help="VideoSegmentation JSON that seeds the annotations",
    )
    parser.add_argument(
        "--metadata",
        help="JSON object, or a path to a JSON file, with user metadata",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing archive"
    )
    args = parser.parse_args()

    if (args.video is None) == (args.frame_folder is None):
        parser.error("give a video path or --frame_folder, exactly one of the two")
    if args.video is not None and not os.path.isfile(args.video):
        parser.error(f"video not found: {args.video}")
    if args.frame_folder is not None and not os.path.isdir(args.frame_folder):
        parser.error(f"frame folder not found: {args.frame_folder}")
    if args.annotation and not os.path.isfile(args.annotation):
        parser.error(f"annotation not found: {args.annotation}")

    metadata = _read_metadata(args.metadata, parser)
    source = args.video or args.frame_folder or ""
    name = project_name_for(args.name, os.path.basename(source.rstrip("/\\")))
    output_dir = args.output_dir or get_settings().projects_dir
    ensure_dir(output_dir)
    output = os.path.join(output_dir, name + PROJECT_EXTENSION)
    if os.path.exists(output) and not args.overwrite:
        parser.error(f"{output} already exists (use --overwrite to replace it)")

    def report(done: int, total: int | None) -> None:
        if done % 100 == 0 or (total is not None and done == total):
            suffix = f"/{total}" if total is not None else ""
            print(f"  {done}{suffix} frames written", file=sys.stderr)

    result = create_project(
        video_path=args.video,
        frame_folder=args.frame_folder,
        # `frame_step` is explicit and wins; 0 means "keep the source rate".
        target_fps=None if args.frame_step else (args.target_fps or None),
        metadata=metadata,
        annotation_path=args.annotation,
        original_fps=args.original_fps,
        output_path=output,
        name=name,
        mode=args.mode,
        frame_step=args.frame_step,
        jpeg_quality=args.jpeg_quality,
        max_frames=args.max_frames,
        progress=report,
    )
    print(
        f"Wrote {result.output_zip}: mode={result.mode}, source={result.source}, "
        f"{result.frame_count} frames, {result.width}x{result.height} @ "
        f"{result.fps:.3f} fps (step {result.frame_step})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

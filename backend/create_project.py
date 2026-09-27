"""Create a Video Segmenter project from a video file."""

from __future__ import annotations

import argparse
import os
import sys

from project_builder import (
    DEFAULT_JPEG_QUALITY,
    MODE_DESCRIPTIONS,
    MODES,
    build_project_from_video,
    project_name_for,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bundle one video into a reviewer project archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {mode}: {desc}" for mode, desc in MODE_DESCRIPTIONS.items()),
    )
    parser.add_argument("video", help="Path to the source video file")
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODES,
        help="Segmentation mode.",
    )
    parser.add_argument("--name", help="Project name")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for the archive when --output is not given",
    )
    parser.add_argument("--frame_step", type=int, default=1, help="Keep every N-th frame")
    parser.add_argument(
        "--jpeg_quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality 1-100"
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Cap on frames written")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing archive"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        parser.error(f"video not found: {args.video}")

    name = project_name_for(args.name, args.video)
    output = os.path.join(args.output_dir, name + ".zip")
    if os.path.exists(output) and not args.overwrite:
        parser.error(f"{output} already exists (use --overwrite to replace it)")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    def report(done: int, total: int | None) -> None:
        if done % 100 == 0 or (total is not None and done == total):
            suffix = f"/{total}" if total is not None else ""
            print(f"  {done}{suffix} frames written", file=sys.stderr)

    result = build_project_from_video(
        args.video,
        output,
        mode=args.mode,
        name=name,
        frame_step=args.frame_step,
        jpeg_quality=args.jpeg_quality,
        progress=report,
    )
    print(
        f"Wrote {result.output_zip}: mode={result.mode}, {result.frame_count} frames, "
        f"{result.width}x{result.height} @ {result.fps:.3f} fps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

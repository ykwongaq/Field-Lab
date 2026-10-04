#!/usr/bin/env python
"""Standalone builder: pack a frame folder + annotation file into a `.project`.

Unlike ``src/cli/create_project.py`` (which also accepts videos), this is the
minimal, frame-only path: point it at a folder of frames and the
VideoSegmentation annotation that describes them, and it writes the same
``.project`` archive the rest of the app reads.

The archive layout is *not* re-implemented here — it is delegated to
``src.projects.builder.create_project``, so a project made by this script is the
same one the API and the browser wizard produce. The annotation file is checked
with the very loader the builder uses, so "valid" means the same thing
everywhere.

Usage:
    python create_project.py \\
        --frame_folder path/to/frames \\
        --annotation_file path/to/annotation.json \\
        --output_path out/my_clip.project
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow `python backend/create_project.py ...` from any working directory: the
# `src` package lives next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.errors import InvalidRequest
from src.projects.builder import MODES, create_project, load_annotation_dataset
from src.projects.layout import PROJECT_EXTENSION, select_frame_files


def validate_annotation(annotation_file: str, frame_folder: str) -> dict:
    """Validate the annotation file, then sanity-check it against the frames.

    Structural validation reuses ``load_annotation_dataset`` — the exact loader
    ``create_project`` calls — so any file accepted here is accepted by the whole
    pipeline.

    Returns the parsed dataset. The frame-name comparison is only a warning: the
    builder deliberately rewrites the frame-describing fields to match what was
    actually written, so a mismatch is not fatal and usually just means the
    annotation was made for a differently resampled copy of the clip.
    """
    if not os.path.isfile(annotation_file):
        raise InvalidRequest(f"Annotation file not found: {annotation_file!r}")

    dataset = load_annotation_dataset(annotation_file)

    try:
        folder_frames = select_frame_files(frame_folder)
    except InvalidRequest:
        return dataset  # let create_project report the folder problem itself

    videos = dataset.get("videos")
    recorded = None
    if isinstance(videos, list) and videos and isinstance(videos[0], dict):
        names = videos[0].get("file_names")
        if isinstance(names, list):
            recorded = names

    if recorded is not None and list(recorded) != list(folder_frames):
        print(
            "warning: the annotation's file_names do not match the frame folder "
            f"({len(recorded)} recorded vs {len(folder_frames)} found); the "
            "frame-describing fields will be rewritten to match the archive.",
            file=sys.stderr,
        )
    return dataset


def main(args: argparse.Namespace) -> int:
    try:
        validate_annotation(args.annotation_file, args.frame_folder)

        output = args.output_path
        if not os.path.splitext(output)[1]:
            output += PROJECT_EXTENSION

        result = create_project(
            frame_folder=args.frame_folder,
            annotation_path=args.annotation_file,
            output_path=output,
            name=args.name,
            mode=args.mode,
            original_fps=args.original_fps,
            frame_step=args.frame_step,
            # Frames are copied verbatim; None keeps the source rate (step 1).
            target_fps=None,
        )
    except InvalidRequest as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {result.output_zip}: mode={result.mode}, "
        f"{result.frame_count} frames, {result.width}x{result.height} @ "
        f"{result.fps:.3f} fps (step {result.frame_step})"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pack a frame folder + annotation file into a .project archive.",
    )
    parser.add_argument(
        "--frame_folder",
        type=str,
        required=True,
        help="Folder holding the frames (images).",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="VideoSegmentation JSON that drives the annotations.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to write the .project archive.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="instance",
        choices=MODES,
        help="Segmentation mode (default: instance).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Project name (default: the frame folder's name).",
    )
    parser.add_argument(
        "--original_fps",
        type=float,
        default=None,
        help="Frame rate of the frames, if known.",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=None,
        help="Keep every N-th frame (default: keep all).",
    )

    raise SystemExit(main(parser.parse_args()))

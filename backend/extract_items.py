#!/usr/bin/env python
"""Extract frames, annotations and/or mask visualizations from `.project` files.

`--input_file` is either one `.project` archive or a folder that is searched
recursively for them. Every project gets its own folder under `--output_folder`,
mirroring its path relative to the input folder (minus the `.project`
extension):

    data/siteA/clip_001.project  ->  out/siteA/clip_001/
                                         frames/           (--frames)
                                         annotation.json   (--annotations)
                                         metadata.json     (--annotations, if present)
                                         visualizations/   (--visualizations)

At least one item flag is required. Items that are requested are overwritten;
items that are not requested are left alone, so the script can be re-run to add
another item without redoing the rest.

The frames written are the ones the masks are aligned to: a project that carries
`frames/` has them copied out under their original names, and a video-only
project is decoded with ffmpeg at the rate the project recorded — the same way
the app does when it opens the project.

Usage:
    python extract_items.py --input_file data --output_folder out \\
        --frames --annotations --visualizations
"""

from __future__ import annotations

import argparse
import colorsys
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Allow `python backend/extract_items.py ...` from any working directory: the
# `src` package lives next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.config import get_settings
from src.domain.extract import extract_frames
from src.domain.rle import decode_rle
from src.projects.layout import (
    ANNOTATION_ENTRY,
    DEFAULT_FPS_FALLBACK,
    FRAMES_DIR,
    IMAGE_EXTENSIONS,
    METADATA_ENTRY,
    PROJECT_EXTENSION,
    VIDEO_DIR,
    natural_key,
    sanitize_name,
)

FRAMES_OUT = "frames"
VIS_OUT = "visualizations"
MASK_ALPHA = 0.45
_LEGACY_ANNOTATION_PREFIX = "annotations/"


# --------------------------------------------------------------------- inputs


def find_projects(input_path: str) -> Tuple[str, List[str]]:
    """`(root, projects)`: the folder paths are relative to, and the archives."""
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return os.path.dirname(input_path), [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input not found: {input_path!r}")

    found: List[str] = []
    for directory, subdirs, files in os.walk(input_path):
        subdirs.sort(key=natural_key)
        for name in sorted(files, key=natural_key):
            if name.lower().endswith(PROJECT_EXTENSION):
                found.append(os.path.join(directory, name))
    return input_path, found


def output_dir_for(project: str, root: str, output_folder: str) -> str:
    relative = os.path.relpath(project, root)
    stem = os.path.splitext(relative)[0]
    return os.path.join(output_folder, stem)


# -------------------------------------------------------------------- archive


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _entries_under(archive: zipfile.ZipFile, prefix: str) -> List[str]:
    return [
        name
        for name in archive.namelist()
        if name.startswith(prefix + "/")
        and not name.endswith("/")
        and not os.path.basename(name).startswith(".")
    ]


def _annotation_entry(archive: zipfile.ZipFile) -> Optional[str]:
    names = archive.namelist()
    if ANNOTATION_ENTRY in names:
        return ANNOTATION_ENTRY
    legacy = sorted(
        name
        for name in names
        if name.lower().startswith(_LEGACY_ANNOTATION_PREFIX)
        and name.lower().endswith(".json")
        and name.count("/") == 1
    )
    return legacy[0] if legacy else None


def _read_dataset(archive: zipfile.ZipFile) -> Dict[str, Any]:
    entry = _annotation_entry(archive)
    if entry is None:
        return {}
    try:
        with archive.open(entry) as handle:
            data = json.load(handle)
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _video_record(dataset: Dict[str, Any]) -> Dict[str, Any]:
    videos = dataset.get("videos")
    record = videos[0] if isinstance(videos, list) and videos else {}
    return record if isinstance(record, dict) else {}


# --------------------------------------------------------------------- frames


@dataclass
class Frames:
    """The frames the annotation is aligned to, on disk, in playback order."""

    directory: str
    names: List[str]
    source: str  # "frames" | "video"


def _ordered_frame_entries(
    archive: zipfile.ZipFile, entries: List[str], recorded: List[str]
) -> List[Tuple[str, str]]:
    """`(name, entry)` pairs in playback order, exactly as the app's loader."""
    by_name = {
        os.path.basename(entry): entry
        for entry in entries
        if os.path.splitext(entry)[1].lower() in IMAGE_EXTENSIONS
    }
    if recorded and len(recorded) == len(by_name) and set(recorded) == set(by_name):
        ordered = list(recorded)
    else:
        ordered = sorted(by_name, key=natural_key)
    return [(name, by_name[name]) for name in ordered]


def materialise_frames(
    archive: zipfile.ZipFile, record: Dict[str, Any], destination: str
) -> Frames:
    """Write the project's frames to `destination` (copied or decoded)."""
    recorded = [
        str(name)
        for name in (record.get("file_names") or [])
        if isinstance(name, str) and name
    ]
    os.makedirs(destination, exist_ok=True)

    frame_entries = _entries_under(archive, FRAMES_DIR)
    if frame_entries:
        pairs = _ordered_frame_entries(archive, frame_entries, recorded)
        if not pairs:
            raise ValueError("The archive's frames/ folder holds no images.")
        for name, entry in pairs:
            with archive.open(entry) as source, open(
                os.path.join(destination, name), "wb"
            ) as sink:
                shutil.copyfileobj(source, sink)
        return Frames(destination, [name for name, _ in pairs], "frames")

    video_entries = _entries_under(archive, VIDEO_DIR)
    if not video_entries:
        raise ValueError("The archive holds neither a frames/ folder nor a video/.")

    fps = (
        _number(record.get("target_fps"))
        or _number(record.get("fps"))
        or DEFAULT_FPS_FALLBACK
    )
    max_size_raw = _number(record.get("max_size"))
    max_size = int(max_size_raw) if max_size_raw and max_size_raw >= 1 else None
    settings = get_settings()

    entry = sorted(video_entries)[0]
    with tempfile.TemporaryDirectory(prefix="extract_items_") as scratch:
        video_path = os.path.join(
            scratch, sanitize_name(os.path.basename(entry), fallback="source.mp4")
        )
        with archive.open(entry) as source, open(video_path, "wb") as sink:
            shutil.copyfileobj(source, sink)
        names = extract_frames(
            video_path,
            destination,
            fps=fps,
            jpeg_quality=settings.frames_jpeg_quality,
            max_size=max_size,
            ffmpeg=settings.ffmpeg_bin,
            scratch_dir=os.path.join(scratch, "raw"),
        )

    if recorded and recorded != names:
        print(
            f"    warning: decoded {len(names)} frames but the annotation records "
            f"{len(recorded)}; masks may not line up."
        )
    return Frames(destination, names, "video")


# ---------------------------------------------------------------- annotations


@dataclass
class Tracklet:
    object_id: int
    label: str
    segmentations: List[Optional[Dict[str, Any]]] = field(default_factory=list)


def _category_name(category: Dict[str, Any]) -> Optional[str]:
    """The name the app shows: common name, else the most specific rank."""
    for key in (
        "common_name",
        "species",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "kingdom",
        "name",
    ):
        value = category.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _categories(dataset: Dict[str, Any]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for category in dataset.get("categories") or []:
        if isinstance(category, dict) and isinstance(category.get("id"), int):
            name = _category_name(category)
            if name:
                names[category["id"]] = name
    return names


def read_tracklets(
    dataset: Dict[str, Any], record: Dict[str, Any], frame_count: int
) -> List[Tracklet]:
    """Instance tracklets, accepting both the app's and per-frame COCO layouts."""
    categories = _categories(dataset)
    video_id = record.get("id")
    rows = [
        row
        for row in dataset.get("annotations") or []
        if isinstance(row, dict)
        and (video_id is None or row.get("video_id") in (None, video_id))
    ]

    def label_of(row: Dict[str, Any], object_id: int) -> str:
        name = categories.get(row.get("category_id"))  # type: ignore[arg-type]
        phrase = row.get("noun_phrase")
        if not name and isinstance(phrase, str) and phrase.strip():
            name = phrase.strip()
        return f"{name or 'object'} #{object_id}"

    per_frame = any(
        isinstance(row.get("image_id"), int)
        and not isinstance(row.get("segmentations"), list)
        for row in rows
    )
    tracklets: List[Tracklet] = []

    if not per_frame:
        for index, row in enumerate(rows):
            segs = row.get("segmentations")
            if not isinstance(segs, list):
                continue
            object_id = row.get("object_id", row.get("id", index + 1))
            tracklets.append(Tracklet(object_id, label_of(row, object_id), segs))
        return tracklets

    grouped: Dict[Any, Tracklet] = {}
    for row in rows:
        object_id = row.get("object_id", row.get("id"))
        tracklet = grouped.get(object_id)
        if tracklet is None:
            tracklet = Tracklet(
                object_id, label_of(row, object_id), [None] * frame_count
            )
            grouped[object_id] = tracklet
        frame = row.get("image_id")
        seg = row.get("segmentation")
        if seg is None and isinstance(row.get("segmentations"), list):
            seg = (row["segmentations"] or [None])[0]
        if isinstance(frame, int) and 0 <= frame < frame_count and seg:
            tracklet.segmentations[frame] = seg
    return list(grouped.values())


def write_annotations(archive: zipfile.ZipFile, out_dir: str) -> bool:
    """Copy `annotation.json` (and `metadata.json`) out verbatim."""
    wrote = False
    entry = _annotation_entry(archive)
    targets = [(entry, ANNOTATION_ENTRY)] if entry else []
    if METADATA_ENTRY in archive.namelist():
        targets.append((METADATA_ENTRY, METADATA_ENTRY))
    for source_name, target_name in targets:
        with archive.open(source_name) as source, open(
            os.path.join(out_dir, target_name), "wb"
        ) as sink:
            shutil.copyfileobj(source, sink)
        wrote = True
    return wrote


# ------------------------------------------------------------- visualizations


def colour_for(key: int) -> Tuple[int, int, int]:
    """A stable, well-spread colour per id (golden-ratio hue walk)."""
    hue = (key * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _contour(mask: np.ndarray) -> np.ndarray:
    """Pixels on the mask's edge (mask minus its 4-neighbour erosion)."""
    padded = np.pad(mask, 1, constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    edge = mask & ~eroded
    # Thicken to ~2px so it stays visible on large frames.
    thick = edge.copy()
    thick[1:, :] |= edge[:-1, :]
    thick[:, 1:] |= edge[:, :-1]
    return thick


def _decode(seg: Dict[str, Any], height: int, width: int) -> Optional[np.ndarray]:
    if not isinstance(seg, dict) or not seg.get("counts"):
        return None
    size = seg.get("size") or [height, width]
    mask = decode_rle(size, seg["counts"])
    if mask.shape != (height, width):
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST
            )
        ).astype(bool)
    return mask


def _label_maps(
    archive: zipfile.ZipFile, record: Dict[str, Any], frame_count: int
) -> Optional[List[Optional[str]]]:
    """Semantic projects: per-frame label-map PNG entries, else `None`."""
    if record.get("segmentation_mode") != "semantic":
        return None
    maps = record.get("label_maps")
    if not isinstance(maps, list):
        return None
    names = set(archive.namelist())
    return [
        maps[i] if i < len(maps) and isinstance(maps[i], str) and maps[i] in names
        else None
        for i in range(frame_count)
    ]


def _font(height: int) -> ImageFont.ImageFont:
    size = max(12, height // 40)
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_visualizations(
    archive: zipfile.ZipFile,
    dataset: Dict[str, Any],
    record: Dict[str, Any],
    frames: Frames,
    out_dir: str,
) -> int:
    """Draw every frame with its masks overlaid. Returns the tracklet count."""
    os.makedirs(out_dir, exist_ok=True)
    count = len(frames.names)
    tracklets = read_tracklets(dataset, record, count)
    label_maps = _label_maps(archive, record, count)
    categories = _categories(dataset)
    font: Optional[ImageFont.ImageFont] = None

    for index, name in enumerate(frames.names):
        with Image.open(os.path.join(frames.directory, name)) as image:
            base = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = base.shape[:2]
        if font is None:
            font = _font(height)
        canvas = base.copy()
        labels: List[Tuple[Tuple[int, int], str, Tuple[int, int, int]]] = []

        layers: List[Tuple[np.ndarray, Tuple[int, int, int], str]] = []
        if label_maps is not None:
            entry = label_maps[index]
            if entry:
                with Image.open(io.BytesIO(archive.read(entry))) as png:
                    ids = np.asarray(png.convert("L").resize((width, height), Image.NEAREST))
                for class_id in np.unique(ids):
                    if class_id == 0:
                        continue
                    text = categories.get(int(class_id), f"class {class_id}")
                    layers.append((ids == class_id, colour_for(int(class_id)), text))
        else:
            for tracklet in tracklets:
                if index >= len(tracklet.segmentations):
                    continue
                mask = _decode(tracklet.segmentations[index], height, width)
                if mask is None or not mask.any():
                    continue
                key = tracklet.object_id if isinstance(tracklet.object_id, int) else hash(tracklet.object_id)
                layers.append((mask, colour_for(key), tracklet.label))

        for mask, colour, text in layers:
            tint = np.array(colour, dtype=np.float32)
            canvas[mask] = canvas[mask] * (1 - MASK_ALPHA) + tint * MASK_ALPHA
            canvas[_contour(mask)] = tint
            ys, xs = np.nonzero(mask)
            labels.append(((int(xs.min()), int(ys.min())), text, colour))

        result = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        if labels:
            draw = ImageDraw.Draw(result)
            for (x, y), text, colour in labels:
                left, top, right, bottom = draw.textbbox((x, y), text, font=font)
                y_shift = min(y, bottom - top + 4)  # sit above the mask when possible
                box = (left, top - y_shift, right + 4, bottom - y_shift + 2)
                draw.rectangle(box, fill=colour)
                draw.text((x + 2, y - y_shift), text, fill=(0, 0, 0), font=font)

        stem = os.path.splitext(name)[0]
        result.save(os.path.join(out_dir, stem + ".jpg"), quality=95)

    return len(tracklets) if label_maps is None else len(categories)


# ----------------------------------------------------------------------- main


def _replace_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def process_project(project: str, out_dir: str, args: argparse.Namespace) -> str:
    """Extract the requested items of one project. Returns a summary line."""
    if not zipfile.is_zipfile(project):
        raise ValueError("not a ZIP archive (expected a .project)")
    os.makedirs(out_dir, exist_ok=True)
    details: List[str] = []

    with zipfile.ZipFile(project) as archive:
        dataset = _read_dataset(archive)
        record = _video_record(dataset)

        if args.annotations:
            for name in (ANNOTATION_ENTRY, METADATA_ENTRY):
                target = os.path.join(out_dir, name)
                if os.path.exists(target):
                    os.remove(target)
            if not write_annotations(archive, out_dir):
                details.append("no annotation.json in archive")

        frames: Optional[Frames] = None
        scratch: Optional[tempfile.TemporaryDirectory] = None
        try:
            if args.frames:
                frames_dir = os.path.join(out_dir, FRAMES_OUT)
                _replace_dir(frames_dir)
                frames = materialise_frames(archive, record, frames_dir)
            elif args.visualizations:
                # Frames are needed to draw on, but were not asked for.
                scratch = tempfile.TemporaryDirectory(prefix="extract_items_frames_")
                frames = materialise_frames(archive, record, scratch.name)

            if frames is not None:
                details.insert(
                    0,
                    f"{len(frames.names)} frames"
                    + (" decoded from video" if frames.source == "video" else ""),
                )

            if args.visualizations and frames is not None:
                vis_dir = os.path.join(out_dir, VIS_OUT)
                _replace_dir(vis_dir)
                tracks = write_visualizations(archive, dataset, record, frames, vis_dir)
                details.append(f"{tracks} tracklets")
        finally:
            if scratch is not None:
                scratch.cleanup()

    return ", ".join(details)


def main(args: argparse.Namespace) -> int:
    try:
        root, projects = find_projects(args.input_file)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not projects:
        print(f"error: no {PROJECT_EXTENSION} files found in {args.input_file!r}", file=sys.stderr)
        return 1

    output_folder = os.path.abspath(args.output_folder)
    failures: List[Tuple[str, str]] = []
    total = len(projects)

    for position, project in enumerate(projects, start=1):
        relative = os.path.relpath(project, root)
        out_dir = output_dir_for(project, root, output_folder)
        shown = os.path.join(args.output_folder, os.path.relpath(out_dir, output_folder))
        print(f"[{position}/{total}] {relative} -> {shown}", flush=True)
        try:
            summary = process_project(project, out_dir, args)
            if summary:
                print(f"    {summary}")
        except Exception as exc:  # noqa: BLE001 - one bad project must not stop the batch
            print(f"    failed: {exc}")
            failures.append((relative, str(exc)))

    print(f"Done: {total - len(failures)} ok, {len(failures)} failed.")
    for relative, reason in failures:
        print(f"  - {relative}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames, annotations and/or visualizations from .project files.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="A .project file, or a folder searched recursively for them.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Where to write one folder per project (mirroring the input layout).",
    )
    parser.add_argument(
        "--frames",
        action="store_true",
        help="Write the frames the masks are aligned to (video projects are decoded with ffmpeg).",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        help="Write annotation.json (and metadata.json if present).",
    )
    parser.add_argument(
        "--visualizations",
        action="store_true",
        help="Write every frame with its masks drawn on top.",
    )
    args = parser.parse_args()
    if not (args.frames or args.annotations or args.visualizations):
        parser.error(
            "nothing to extract: pick at least one of "
            "--frames, --annotations, --visualizations"
        )
    raise SystemExit(main(args))

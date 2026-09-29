"""Bundle a directory of frame folders (+ annotations) into project archives."""

import argparse
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

from src.projects.builder import (
    ANNOTATION_ENTRY,
    FRAMES_DIR,
    MODES,
    VIDEO_DIR,
    VIDEO_EXTENSIONS,
    build_annotation_dataset,
    write_json_entry,
)

# Compression name -> (zipfile constant, compresslevel).
# For ZIP_STORED and ZIP_LZMA the compresslevel is ignored by zipfile.
COMPRESSION_OPTIONS = {
    "stored": (zipfile.ZIP_STORED, None),
    "deflated": (zipfile.ZIP_DEFLATED, 9),
    "bzip2": (zipfile.ZIP_BZIP2, 9),
    "lzma": (zipfile.ZIP_LZMA, None),
}


def find_video_file(video_dataset, clip_name):
    """Return the video file for `clip_name` inside `video_dataset`, if any."""
    if not video_dataset:
        return None
    for ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(video_dataset, clip_name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def read_dataset(annotation_file):
    """Load the annotation JSON; `None` when it cannot be parsed."""
    try:
        with open(annotation_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def create_clip_zip(task):
    """Build a single project zip containing the clip's frames and annotation.

    Layout inside each zip:
        frames/<image files>
        annotation.json                (only if an annotation exists)

    Returns (clip_name, has_annotation, error_message_or_None).
    """
    frame_dir, annotation_file, video_file, output_zip, compression_name, mode = task
    compression, compresslevel = COMPRESSION_OPTIONS[compression_name]
    clip_name = os.path.basename(frame_dir)

    frames = []
    for name in sorted(os.listdir(frame_dir)):
        path = os.path.join(frame_dir, name)
        if os.path.isfile(path):
            frames.append((path, f"{FRAMES_DIR}/{name}"))

    has_annotation = annotation_file is not None and os.path.isfile(annotation_file)
    has_video = video_file is not None and os.path.isfile(video_file)

    try:
        dataset = read_dataset(annotation_file) if has_annotation else None
        record = (dataset or {}).get("videos") or [{}]
        record = record[0] if isinstance(record[0], dict) else {}
        fps = record.get("fps") if isinstance(record.get("fps"), (int, float)) else None
        width = record.get("width")
        height = record.get("height")
        if (not width or not height) and frames:
            probe = cv2.imread(frames[0][0])
            if probe is not None:
                height, width = probe.shape[:2]

        video_entry = None
        if has_video:
            video_entry = (
                f"{VIDEO_DIR}/{clip_name}{os.path.splitext(video_file)[1].lower()}"
            )

        with zipfile.ZipFile(
            output_zip,
            "w",
            compression=compression,
            compresslevel=compresslevel,
            allowZip64=True,
        ) as zf:
            for path, arcname in frames:
                zf.write(
                    path,
                    arcname=arcname,
                    compress_type=compression,
                    compresslevel=compresslevel,
                )

            annotation_entry = ANNOTATION_ENTRY
            if dataset is not None:
                if record.get("segmentation_mode") not in MODES:
                    record["segmentation_mode"] = mode
                elif record["segmentation_mode"] != mode:
                    print(
                        f"[{clip_name}] keeps its existing mode "
                        f"{record['segmentation_mode']!r} (ignoring --mode {mode})"
                    )
                if video_entry and not record.get("video_file"):
                    record["video_file"] = video_entry
                write_json_entry(zf, annotation_entry, dataset)
            else:
                write_json_entry(
                    zf,
                    annotation_entry,
                    build_annotation_dataset(
                        name=clip_name,
                        mode=mode,
                        file_names=[arcname.split("/", 1)[1] for _, arcname in frames],
                        fps=float(fps) if fps else 25.0,
                        width=int(width or 0),
                        height=int(height or 0),
                    ),
                )

            if has_video:
                zf.write(
                    video_file, arcname=video_entry, compress_type=zipfile.ZIP_STORED
                )

    except Exception as exc:
        if os.path.exists(output_zip):
            os.remove(output_zip)
        return clip_name, has_annotation, str(exc)

    return clip_name, has_annotation, None


def main(args):
    frame_dataset = args.frame_dataset
    annotation_dataset = args.annotation_dataset
    video_dataset = args.video_dataset
    output_dataset = args.output_dataset
    compression = args.compression
    mode = args.mode

    os.makedirs(output_dataset, exist_ok=True)

    frame_dirs = sorted(
        os.path.join(frame_dataset, name)
        for name in os.listdir(frame_dataset)
        if os.path.isdir(os.path.join(frame_dataset, name))
    )

    tasks = []
    skipped = 0
    for frame_dir in frame_dirs:
        name = os.path.basename(frame_dir)
        annotation_file = os.path.join(annotation_dataset, name + ".json")
        if not os.path.isfile(annotation_file):
            annotation_file = None

        output_zip = os.path.join(output_dataset, name + ".zip")
        if os.path.exists(output_zip):
            skipped += 1
            continue

        video_file = find_video_file(video_dataset, name)
        tasks.append(
            (frame_dir, annotation_file, video_file, output_zip, compression, mode)
        )

    if not tasks:
        print(
            f"Nothing to do: {len(frame_dirs)} clip folders found, {skipped} already zipped."
        )
        return

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    total = len(tasks)
    print(f"Found {len(frame_dirs)} clip folders ({skipped} already zipped).")
    print(
        f"Zipping {total} clips with compression={compression} using {workers} worker(s) ..."
    )

    done = 0
    missing_annotations = 0
    errors = 0

    def report(name, has_annotation, error):
        nonlocal done, missing_annotations, errors
        done += 1
        if error is not None:
            errors += 1
            print(f"  ERROR {name}: {error}")
        elif not has_annotation:
            missing_annotations += 1
        if done % 100 == 0 or done == total:
            print(f"  Progress: {done}/{total} clips zipped")

    if workers <= 1:
        for task in tasks:
            name, has_annotation, error = create_clip_zip(task)
            report(name, has_annotation, error)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(create_clip_zip, task) for task in tasks]
            for future in as_completed(futures):
                name, has_annotation, error = future.result()
                report(name, has_annotation, error)

    print(
        f"Done. {total} zips written to {output_dataset} "
        f"({errors} errors, {missing_annotations} clips without an annotation)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bundle each clip's frames (and its annotation, if present) "
        "into a single compressed zip file."
    )
    parser.add_argument(
        "--frame_dataset", type=str, default="/home/davidwong/Downloads/frames"
    )
    parser.add_argument(
        "--annotation_dataset",
        type=str,
        default="/home/davidwong/Downloads/annotations",
    )
    parser.add_argument(
        "--video_dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        default="/home/davidwong/Downloads/output_projects",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=MODES,
        help="Segmentation mode written as `segmentation_mode` on the video record ",
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=sorted(COMPRESSION_OPTIONS),
        default="deflated",
        help="Compression method. Defaults to 'deflated' because the web reviewer can "
        "only read STORED (0) and DEFLATE (8) entries. 'lzma' produces smaller files "
        "but the reviewer cannot open them (Unsupported compression method 14); "
        "'bzip2' is likewise unsupported. 'deflated' (level 9) is the recommended "
        "balance of size vs. speed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel processes to use (default: number of CPUs).",
    )

    args = parser.parse_args()
    main(args)

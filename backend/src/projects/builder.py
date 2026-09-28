"""Build a reviewer project archive from a video file.

The archive layout produced here is a contract with the browser reader
(`frontend/src/lib/zip.ts`):

    video/<name><ext>          the source video (stored, not re-compressed)
    frames/000000.jpg ...      every kept frame, STORED so the browser can
                               inflate it without an LZMA-capable reader
    annotations/<name>.json    pycocotools VideoSegmentation skeleton
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

import cv2

from src.core.errors import InvalidRequest

VIDEO_DIR = "video"
FRAMES_DIR = "frames"
ANNOTATIONS_DIR = "annotations"
FRAME_NAME_PATTERN = "{:06d}.jpg"

ProjectMode = Literal["instance", "semantic"]
MODES: tuple[ProjectMode, ...] = ("instance", "semantic")
MODE_DESCRIPTIONS: Dict[str, str] = {
    "instance": (
        "Every tracked object gets its own mask and identity (tracklets); "
        "two fish of the same species are two objects."
    ),
    "semantic": (
        "Pixels are labelled by class only; all fish of one species share "
        "a single mask per frame."
    ),
}
DEFAULT_MODE: ProjectMode = "instance"  # what archives without the field mean

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
)

DEFAULT_JPEG_QUALITY = 100
DEFAULT_FPS_FALLBACK = 25.0

ProgressCallback = Callable[[int, Optional[int]], None]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def validate_mode(mode: str) -> ProjectMode:
    if mode not in MODES:
        raise InvalidRequest(
            f"Unknown project mode {mode!r}; expected one of {', '.join(MODES)}."
        )
    return mode


def sanitize_name(
    raw: str, *, from_filename: bool = False, fallback: str = "project"
) -> str:
    text = raw or ""
    if from_filename:
        text = os.path.splitext(os.path.basename(text.replace("\\", "/")))[0]
    cleaned = _SAFE_NAME.sub("_", text).strip("._-")
    return cleaned or fallback


def project_name_for(name: Optional[str], filename: str) -> str:
    if name and name.strip():
        return sanitize_name(name)
    return sanitize_name(filename, from_filename=True)


def frame_name(index: int) -> str:
    return FRAME_NAME_PATTERN.format(index)


def build_annotation_dataset(
    *,
    name: str,
    mode: str,
    file_names: List[str],
    fps: float,
    width: int,
    height: int,
    source_video: Optional[str] = None,
    video_entry: Optional[str] = None,
) -> Dict[str, Any]:
    mode = validate_mode(mode)
    video: Dict[str, Any] = {
        "id": 1,
        "video_name": name,
        "file_names": file_names,
        "length": len(file_names),
        "height": height,
        "width": width,
        "fps": fps,
        "segmentation_mode": mode,
        "original_video": source_video,
        "video_file": video_entry,
        "start_frame": 0,
        "end_frame": max(0, len(file_names) - 1),
        "status": "unannotated",
    }
    if mode == "semantic":
        video["label_maps"] = [None] * len(file_names)
    return {"videos": [video], "annotations": [], "categories": []}


def write_json_entry(zf: zipfile.ZipFile, arcname: str, payload: Any) -> None:
    zf.writestr(
        arcname,
        json.dumps(payload, ensure_ascii=False, indent=2),
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    )


class VideoOpenError(InvalidRequest):
    """Raised when OpenCV cannot open or decode the source video."""


@dataclass
class ProjectBuildResult:
    name: str
    mode: str
    output_zip: str
    frame_count: int
    fps: float
    width: int
    height: int
    video_entry: str
    annotation_entry: str


def probe_video(
    video_path: str, label: Optional[str] = None
) -> tuple[float, int, int, int]:
    """Return (fps, width, height, estimated_frame_count) for a video."""
    label = label or os.path.basename(video_path)
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise VideoOpenError(
                f"Could not open video {label!r} (unsupported container or codec)."
            )
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        estimate = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if fps <= 0 or fps != fps:  # 0 or NaN
        fps = DEFAULT_FPS_FALLBACK
    return fps, width, height, estimate


def build_project_from_video(
    video_path: str,
    output_zip: str,
    *,
    mode: str,
    name: Optional[str] = None,
    frame_step: int = 1,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    max_frames: Optional[int] = None,
    source_filename: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
) -> ProjectBuildResult:
    """Encode every kept frame as JPEG and write the project archive.

    Frames are STORED (JPEG is already compressed) while the annotation JSON is
    DEFLATED. The archive is written to `<output_zip>.part` and moved into place
    only on success, so a crashed run never leaves a half-written project.
    """
    mode = validate_mode(mode)
    if frame_step < 1:
        raise InvalidRequest("frame_step must be >= 1")
    if not 1 <= jpeg_quality <= 100:
        raise InvalidRequest("jpeg_quality must be within 1..100")

    source_filename = source_filename or os.path.basename(video_path)
    project_name = project_name_for(name, source_filename)
    source_fps, width, height, estimate = probe_video(video_path, source_filename)
    fps = source_fps / frame_step
    estimated_out = (estimate + frame_step - 1) // frame_step if estimate else None
    if max_frames is not None and estimated_out is not None:
        estimated_out = min(estimated_out, max_frames)

    video_ext = os.path.splitext(source_filename)[1].lower() or ".mp4"
    video_entry = f"{VIDEO_DIR}/{project_name}{video_ext}"
    annotation_entry = f"{ANNOTATIONS_DIR}/{project_name}.json"

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    file_names: List[str] = []

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise VideoOpenError(
            f"Could not open video {source_filename!r} (unsupported container or codec)."
        )

    tmp_zip = output_zip + ".part"
    try:
        with zipfile.ZipFile(tmp_zip, "w", allowZip64=True) as zf:
            # 1. Frames.
            decoded_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if decoded_index % frame_step == 0:
                    if max_frames is not None and len(file_names) >= max_frames:
                        break
                    ok_enc, buffer = cv2.imencode(".jpg", frame, encode_params)
                    if not ok_enc:
                        raise RuntimeError(
                            f"JPEG encoding failed at frame {decoded_index}"
                        )
                    if not file_names:
                        height, width = frame.shape[:2]
                    arc = frame_name(len(file_names))
                    zf.writestr(
                        f"{FRAMES_DIR}/{arc}",
                        buffer.tobytes(),
                        compress_type=zipfile.ZIP_STORED,
                    )
                    file_names.append(arc)
                    if progress:
                        progress(len(file_names), estimated_out)
                decoded_index += 1
            capture.release()

            if not file_names:
                raise VideoOpenError(
                    f"No frames could be decoded from {source_filename!r} "
                    "(unsupported codec or empty file)."
                )

            # 2. Source video.
            zf.write(video_path, arcname=video_entry, compress_type=zipfile.ZIP_STORED)

            # 3. Annotation.
            dataset = build_annotation_dataset(
                name=project_name,
                mode=mode,
                file_names=file_names,
                fps=fps,
                width=width,
                height=height,
                source_video=source_filename,
                video_entry=video_entry,
            )
            write_json_entry(zf, annotation_entry, dataset)
        os.replace(tmp_zip, output_zip)
    except Exception:
        capture.release()
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        raise

    return ProjectBuildResult(
        name=project_name,
        mode=mode,
        output_zip=output_zip,
        frame_count=len(file_names),
        fps=fps,
        width=width,
        height=height,
        video_entry=video_entry,
        annotation_entry=annotation_entry,
    )

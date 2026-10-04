"""Build a project archive (`.project`, a ZIP) from a video or a frame folder.

The archive layout produced here is a contract with the browser reader
(`frontend/src/lib/zip.ts`):

    video/<name><ext>          the source video, when one was uploaded
                               (STORED, not re-compressed)
    frames/000000.jpg ...      every kept frame, STORED so the browser can
                               inflate it without an LZMA-capable reader
    annotation.json            VideoSegmentation dataset: the caller's annotation
                               file with its video record completed, or a
                               fresh skeleton
    metadata.json              free-form user metadata (always written)

`create_project()` is the entry point. `build_project_from_video()` remains as a
thin wrapper for callers that think in `frame_step` rather than `target_fps`.

This module is the *batch* path (CLI, scripts, `make_projects.py`). Interactive
creation happens in the browser instead — `frontend/src/lib/projectWriter.ts` —
which packs the same layout without uploading anything.

`cv2` is imported inside the functions that need it, so reading this layout (or
building a session from an archive) never requires the image stack.
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

from src.core.config import DEFAULT_TARGET_FPS
from src.core.errors import InvalidRequest
from src.core.storage import ensure_dir
from src.projects.layout import (
    ANNOTATION_ENTRY,
    DEFAULT_FPS_FALLBACK,
    DEFAULT_JPEG_QUALITY,
    FRAMES_DIR,
    METADATA_ENTRY,
    PROJECT_EXTENSION,
    VIDEO_DIR,
    archive_frame_name,
    project_name_for,
    select_frame_files,
)

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

ProgressCallback = Callable[[int, Optional[int]], None]


def validate_mode(mode: str) -> ProjectMode:
    if mode not in MODES:
        raise InvalidRequest(
            f"Unknown project mode {mode!r}; expected one of {', '.join(MODES)}."
        )
    return mode


def _video_record(
    *,
    name: str,
    mode: str,
    file_names: List[str],
    fps: float,
    width: int,
    height: int,
    source_video: Optional[str],
    video_entry: Optional[str],
    original_fps: Optional[float] = None,
    target_fps: Optional[float] = None,
    frame_step: int = 1,
) -> Dict[str, Any]:
    """The single `videos[0]` record every project archive carries.

    `fps` is the effective rate of the frames that were actually written,
    `original_fps` the source rate and `target_fps` what the caller asked for —
    integer decimation rarely lands exactly on the target (25 fps asked for 6
    gives a step of 4 and 6.25 fps), so both are recorded for provenance.
    """
    record: Dict[str, Any] = {
        "id": 1,
        "video_name": name,
        "file_names": file_names,
        "length": len(file_names),
        "height": height,
        "width": width,
        "fps": fps,
        "original_fps": original_fps,
        "target_fps": target_fps,
        "frame_step": frame_step,
        "segmentation_mode": mode,
        "original_video": source_video,
        "video_file": video_entry,
        "start_frame": 0,
        "end_frame": max(0, len(file_names) - 1),
        "status": "unannotated",
    }
    if mode == "semantic":
        record["label_maps"] = [None] * len(file_names)
    return record


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
    original_fps: Optional[float] = None,
    target_fps: Optional[float] = None,
    frame_step: int = 1,
) -> Dict[str, Any]:
    """An empty VideoSegmentation dataset for a freshly created project."""
    mode = validate_mode(mode)
    return {
        "videos": [
            _video_record(
                name=name,
                mode=mode,
                file_names=file_names,
                fps=fps,
                width=width,
                height=height,
                source_video=source_video,
                video_entry=video_entry,
                original_fps=original_fps,
                target_fps=target_fps,
                frame_step=frame_step,
            )
        ],
        "annotations": [],
        "categories": [],
    }


def load_annotation_dataset(path: str) -> Dict[str, Any]:
    """Read a user-supplied VideoSegmentation JSON and check its shape.

    A project holds exactly one clip, so a file describing several videos is
    rejected rather than silently truncated.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise InvalidRequest(
            f"Could not read the annotation file {path!r}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidRequest(
            f"The annotation file {path!r} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise InvalidRequest("The annotation file must contain a JSON object.")
    for key in ("annotations", "categories"):
        if key in data and not isinstance(data[key], list):
            raise InvalidRequest(f"`{key}` in the annotation file must be a list.")
    videos = data.get("videos")
    if videos is None:
        if not any(key in data for key in ("annotations", "categories")):
            raise InvalidRequest(
                "The annotation file does not look like a VideoSegmentation "
                "dataset (expected `videos`, `annotations` and `categories`)."
            )
        return data
    if not isinstance(videos, list):
        raise InvalidRequest("`videos` in the annotation file must be a list.")
    if len(videos) > 1:
        raise InvalidRequest(
            f"The annotation file describes {len(videos)} videos; a project "
            "holds exactly one clip."
        )
    if videos and not isinstance(videos[0], dict):
        raise InvalidRequest("`videos[0]` in the annotation file must be an object.")
    return data


def initialize_annotation_dataset(
    dataset: Optional[Dict[str, Any]],
    *,
    name: str,
    mode: str,
    file_names: List[str],
    fps: float,
    width: int,
    height: int,
    source_video: Optional[str],
    video_entry: Optional[str],
    original_fps: Optional[float],
    target_fps: Optional[float],
    frame_step: int,
) -> Dict[str, Any]:
    """Complete an annotation dataset so it matches the frames in the archive.

    `annotations` and `categories` (and any extra keys on the video record, e.g.
    `scene_id`) are kept as the caller supplied them. The fields that describe
    the frames are overwritten — they have to agree with the archive — while
    `id` and `status` are preserved because annotation `video_id`s point at the
    former and the latter records how far the review got.
    """
    record = _video_record(
        name=name,
        mode=mode,
        file_names=file_names,
        fps=fps,
        width=width,
        height=height,
        source_video=source_video,
        video_entry=video_entry,
        original_fps=original_fps,
        target_fps=target_fps,
        frame_step=frame_step,
    )
    if dataset is None:
        return {"videos": [record], "annotations": [], "categories": []}

    videos = dataset.get("videos")
    if videos:
        existing = videos[0]
        preserved = {
            key: existing[key]
            for key in ("id", "status")
            if key in existing and existing[key] is not None
        }
        existing.update(record)
        existing.update(preserved)
    else:
        dataset["videos"] = [record]
    dataset.setdefault("annotations", [])
    dataset.setdefault("categories", [])
    return dataset


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
    """What `create_project` wrote, and what it decided along the way."""

    name: str
    mode: str
    output_zip: str  # the `.project` file (a ZIP) on disk
    frame_count: int
    fps: float  # effective frame rate of the frames in the archive
    width: int
    height: int
    annotation_entry: str
    video_entry: Optional[str] = None  # None when the source was a frame folder
    original_fps: Optional[float] = None
    target_fps: Optional[float] = None
    frame_step: int = 1
    metadata_entry: str = METADATA_ENTRY
    source: str = "video"  # "video" | "frames"


def probe_video(
    video_path: str, label: Optional[str] = None
) -> tuple[float, int, int, int]:
    """Return (fps, width, height, estimated_frame_count) for a video."""
    import cv2

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


@dataclass
class _FramesWritten:
    """Bookkeeping for the frames that ended up in `frames/`."""

    file_names: List[str]
    width: int
    height: int


def probe_frame_folder(folder: str) -> tuple[int, int, int]:
    """Return (frame_count, width, height) for a folder of frames."""
    import cv2

    names = select_frame_files(folder)
    first = cv2.imread(os.path.join(folder, names[0]))
    if first is None:
        raise InvalidRequest(
            f"The first frame of {folder!r} ({names[0]!r}) could not be decoded."
        )
    height, width = first.shape[:2]
    return len(names), width, height


def plan_resampling(
    source_fps: Optional[float],
    target_fps: Optional[float],
    frame_step: Optional[int],
) -> tuple[int, Optional[float]]:
    """Decide the decimation step and the effective fps of the kept frames.

    `frame_step` wins when given. Otherwise the step is
    `round(source_fps / target_fps)`, which is why the effective rate can differ
    slightly from the requested one (25 fps at a target of 6 gives 4 and 6.25).
    Frames are never invented, so a target above the source rate keeps every
    frame at the source rate.
    """
    if frame_step is not None:
        if frame_step < 1:
            raise InvalidRequest("frame_step must be >= 1")
        return frame_step, (source_fps / frame_step if source_fps else None)
    if target_fps is None:
        return 1, source_fps
    if source_fps is None:
        # A frame folder with no known source rate: trust the requested value.
        return 1, float(target_fps)
    if source_fps <= target_fps:
        return 1, source_fps
    step = max(1, int(round(source_fps / target_fps)))
    return step, source_fps / step


def _write_frames_from_video(
    zf: zipfile.ZipFile,
    video_path: str,
    *,
    step: int,
    max_frames: Optional[int],
    jpeg_quality: int,
    estimate: Optional[int],
    progress: Optional[ProgressCallback],
) -> _FramesWritten:
    """Decode the video and keep every `step`-th frame as a JPEG in `frames/`."""
    import cv2

    total = (estimate + step - 1) // step if estimate else None
    if total is not None and max_frames is not None:
        total = min(total, max_frames)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    label = os.path.basename(video_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise VideoOpenError(
            f"Could not open video {label!r} (unsupported container or codec)."
        )

    file_names: List[str] = []
    width = height = 0
    try:
        decoded_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded_index % step == 0:
                if max_frames is not None and len(file_names) >= max_frames:
                    break
                ok_encoded, buffer = cv2.imencode(".jpg", frame, encode_params)
                if not ok_encoded:
                    raise RuntimeError(f"JPEG encoding failed at frame {decoded_index}")
                if not file_names:
                    height, width = frame.shape[:2]
                arc = archive_frame_name(len(file_names))
                zf.writestr(
                    f"{FRAMES_DIR}/{arc}",
                    buffer.tobytes(),
                    compress_type=zipfile.ZIP_STORED,
                )
                file_names.append(arc)
                if progress:
                    progress(len(file_names), total)
            decoded_index += 1

        if not file_names:
            raise VideoOpenError(
                f"No frames could be decoded from {label!r} "
                "(unsupported codec or empty file)."
            )
    finally:
        capture.release()
    return _FramesWritten(file_names, width, height)


def _write_frames_from_folder(
    zf: zipfile.ZipFile,
    folder: str,
    *,
    step: int,
    max_frames: Optional[int],
    progress: Optional[ProgressCallback],
) -> _FramesWritten:
    """Copy the folder's images into `frames/`, keeping every `step`-th file.

    The files are stored byte-for-byte: they are already images, so re-encoding
    would only cost time and quality. Names are kept, which means `file_names`
    mirrors the folder the caller uploaded.
    """
    import cv2

    available = select_frame_files(folder)
    selected = available[::step]
    if max_frames is not None:
        selected = selected[:max_frames]

    first = cv2.imread(os.path.join(folder, selected[0]))
    if first is None:
        raise InvalidRequest(
            f"The first frame of {folder!r} ({selected[0]!r}) could not be decoded."
        )
    height, width = first.shape[:2]

    file_names: List[str] = []
    for arc in selected:
        with open(os.path.join(folder, arc), "rb") as handle:
            data = handle.read()
        zf.writestr(f"{FRAMES_DIR}/{arc}", data, compress_type=zipfile.ZIP_STORED)
        file_names.append(arc)
        if progress:
            progress(len(file_names), len(selected))
    return _FramesWritten(file_names, width, height)


def create_project(
    video_path: Optional[str] = None,
    frame_folder: Optional[str] = None,
    target_fps: Optional[float] = DEFAULT_TARGET_FPS,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    annotation_path: Optional[str] = None,
    original_fps: Optional[float] = None,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    mode: str = DEFAULT_MODE,
    frame_step: Optional[int] = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    max_frames: Optional[int] = None,
    source_filename: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
) -> ProjectBuildResult:
    """Create a `.project` archive (a ZIP) from a video or a folder of frames.

    Exactly one source is used:

    * `video_path` — frames are decoded from the video, decimated, and written
      as JPEGs; the video itself is stored in the archive as well.
    * `frame_folder` — the folder's images are copied in as-is (no re-encode),
      ordered naturally by file name and decimated the same way.

    `target_fps` is the rate the reviewer should see (see `plan_resampling` for
    how integer decimation maps it onto the real frame rate); pass `None` to keep
    the source rate, or `frame_step` to decimate explicitly — it wins.

    `annotation_path` initialises the annotations from a VideoSegmentation JSON
    (its video record is completed so it matches the frames actually written);
    `metadata` is stored verbatim as `metadata.json`. Both are optional.

    The archive is written to `<output_path>.part` and renamed only on success,
    so an interrupted run never publishes a half-written project, and no scratch
    files are left in the temp directory (callers stage uploads there with
    `core.storage.scratch_dir`, which cleans up after itself).
    """
    if (video_path is None) == (frame_folder is None):
        raise InvalidRequest(
            "Provide exactly one source: `video_path` or `frame_folder`."
        )
    if frame_folder is not None and not os.path.isdir(frame_folder):
        raise InvalidRequest(f"Frame folder not found: {frame_folder!r}")
    if video_path is not None and not os.path.isfile(video_path):
        raise InvalidRequest(f"Video not found: {video_path!r}")
    if annotation_path is not None and not os.path.isfile(annotation_path):
        raise InvalidRequest(f"Annotation file not found: {annotation_path!r}")
    if metadata is not None and not isinstance(metadata, dict):
        raise InvalidRequest("`metadata` must be a dictionary.")
    if target_fps is not None and target_fps <= 0:
        raise InvalidRequest("target_fps must be > 0")
    if not 1 <= jpeg_quality <= 100:
        raise InvalidRequest("jpeg_quality must be within 1..100")
    if max_frames is not None and max_frames < 1:
        raise InvalidRequest("max_frames must be >= 1")
    mode = validate_mode(mode)

    metadata_dict: Dict[str, Any] = dict(metadata or {})
    try:
        json.dumps(metadata_dict, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise InvalidRequest(f"`metadata` is not JSON-serialisable: {exc}") from exc

    # 1. Probe the source for its frame rate (and frame size, as a sanity check).
    if video_path is not None:
        default_source = source_filename or os.path.basename(video_path)
        source_kind = "video"
        probed_fps, _probe_width, _probe_height, estimate = probe_video(
            video_path, default_source
        )
        source_fps: Optional[float] = probed_fps
    else:
        default_source = source_filename or os.path.basename(
            os.path.abspath(frame_folder or "")
        )
        source_kind = "frames"
        source_fps = float(original_fps) if original_fps else None
        estimate = None

    project_name = project_name_for(name, default_source)
    step, effective_fps = plan_resampling(source_fps, target_fps, frame_step)
    fps_out = effective_fps if effective_fps else DEFAULT_FPS_FALLBACK

    extension = os.path.splitext(default_source)[1].lower() or ".mp4"
    video_entry = (
        f"{VIDEO_DIR}/{project_name}{extension}" if video_path is not None else None
    )
    annotation_entry = ANNOTATION_ENTRY

    # 2. Read the caller's annotation file up front: a broken one should fail
    #    before an hour of frame extraction is spent.
    dataset = load_annotation_dataset(annotation_path) if annotation_path else None

    # 3. Resolve where the archive goes.
    if output_path is None:
        output_path = os.path.join(output_dir or ".", project_name + PROJECT_EXTENSION)
    elif not os.path.splitext(output_path)[1]:
        output_path = output_path + PROJECT_EXTENSION
    if os.path.isdir(output_path):
        raise InvalidRequest(f"{output_path!r} is a directory, not a file.")
    ensure_dir(os.path.dirname(os.path.abspath(output_path)))

    # 4. Write the archive: frames, then the source video, the annotation and the
    #    metadata. JPEGs and the video are STORED (already compressed); the two
    #    JSON entries are DEFLATED.
    tmp_zip = output_path + ".part"
    try:
        with zipfile.ZipFile(tmp_zip, "w", allowZip64=True) as zf:
            if video_path is not None:
                written = _write_frames_from_video(
                    zf,
                    video_path,
                    step=step,
                    max_frames=max_frames,
                    jpeg_quality=jpeg_quality,
                    estimate=estimate,
                    progress=progress,
                )
                zf.write(
                    video_path,
                    arcname=f"{VIDEO_DIR}/{project_name}{extension}",
                    compress_type=zipfile.ZIP_STORED,
                )
            else:
                written = _write_frames_from_folder(
                    zf,
                    frame_folder or "",
                    step=step,
                    max_frames=max_frames,
                    progress=progress,
                )

            dataset = initialize_annotation_dataset(
                dataset,
                name=project_name,
                mode=mode,
                file_names=written.file_names,
                fps=fps_out,
                width=written.width,
                height=written.height,
                source_video=default_source if video_path is not None else None,
                video_entry=video_entry,
                original_fps=source_fps,
                target_fps=target_fps,
                frame_step=step,
            )
            write_json_entry(zf, annotation_entry, dataset)
            write_json_entry(zf, METADATA_ENTRY, metadata_dict)
        os.replace(tmp_zip, output_path)
    except Exception:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        raise

    return ProjectBuildResult(
        name=project_name,
        mode=mode,
        output_zip=output_path,
        frame_count=len(written.file_names),
        fps=fps_out,
        width=written.width,
        height=written.height,
        annotation_entry=annotation_entry,
        video_entry=video_entry,
        original_fps=source_fps,
        target_fps=target_fps,
        frame_step=step,
        metadata_entry=METADATA_ENTRY,
        source=source_kind,
    )


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
    """Bundle a video using an explicit `frame_step`.

    Thin wrapper around `create_project` for callers that decimate by step
    instead of asking for a target frame rate.
    """
    return create_project(
        video_path=video_path,
        frame_folder=None,
        target_fps=None,
        metadata=None,
        output_path=output_zip,
        name=name,
        mode=mode,
        frame_step=frame_step,
        jpeg_quality=jpeg_quality,
        max_frames=max_frames,
        source_filename=source_filename,
        progress=progress,
    )

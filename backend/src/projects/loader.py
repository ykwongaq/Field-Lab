"""Materialise an uploaded `.project` archive into a session's frames.

A session's `frames/` directory is the one frame source every reader works from,
so this module's job is to fill it, whatever the archive carried:

* `frames/` entries — copied across as they are, keeping the archive's names and
  the playback order its annotation recorded.
* a `video/` entry — decoded with ffmpeg at the rate the archive asked for, and
  numbered `00000000.jpg` and up.

Nothing here writes back to the archive. The archive is the durable record and
the session is a cache of it, so the result describes what the frames now are;
the caller owns the archive and updates it when it next saves.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import Settings
from src.core.errors import InvalidRequest, UnsupportedMediaType
from src.core.sessions import Session
from src.domain.extract import extract_frames, frame_dimensions, probe_video
from src.projects.layout import (
    ANNOTATION_ENTRY,
    DEFAULT_FPS_FALLBACK,
    FRAMES_DIR,
    IMAGE_EXTENSIONS,
    VIDEO_DIR,
    natural_key,
    sanitize_name,
)

#: Archives written before the annotation moved to the archive root.
_LEGACY_ANNOTATION = re.compile(r"\Aannotations/[^/]+\.json\Z", re.IGNORECASE)


@dataclass
class LoadedProject:
    """What the session now holds, next to what the archive claimed."""

    source: str  # "frames" | "video"
    frame_names: List[str]
    fps: float
    width: int
    height: int
    mode: Optional[str] = None
    original_fps: Optional[float] = None
    video_entry: Optional[str] = None
    archive_name: Optional[str] = None
    annotation_entry: Optional[str] = None
    recorded_frame_names: Optional[List[str]] = None

    @property
    def frame_count(self) -> int:
        return len(self.frame_names)

    @property
    def frame_names_match(self) -> bool:
        """Whether the archive's recorded frame list agrees with what we built.

        For a frame source they agree by construction — the frames *are* the
        archive's frames. For a video source this is the check that re-extracting
        produced the same sequence, so a drift after an ffmpeg change surfaces as
        a mismatch instead of silently shifting every saved mask.
        """
        recorded = self.recorded_frame_names
        if not recorded:
            return True
        return list(recorded) == list(self.frame_names)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _entry_names(archive: zipfile.ZipFile, prefix: str) -> List[str]:
    """Files under `prefix/`, ignoring directory placeholders."""
    return [
        name
        for name in archive.namelist()
        if name.startswith(prefix + "/")
        and not name.endswith("/")
        and not os.path.basename(name).startswith(".")
    ]


def _read_dataset(archive: zipfile.ZipFile) -> Tuple[Dict[str, Any], Optional[str]]:
    """Read the clip's annotation, preferring the root entry.

    A missing or unreadable annotation is not fatal: the frames are what the
    session needs, and everything else in the record is recoverable from the
    file names.
    """
    names = set(archive.namelist())
    entry = ANNOTATION_ENTRY if ANNOTATION_ENTRY in names else None
    if entry is None:
        legacy = sorted(name for name in names if _LEGACY_ANNOTATION.match(name))
        entry = legacy[0] if legacy else None
    if entry is None:
        return {}, None
    try:
        with archive.open(entry) as handle:
            data = json.load(handle)
    except (ValueError, OSError):
        return {}, entry
    return (data if isinstance(data, dict) else {}), entry


def _copy_frames(
    archive: zipfile.ZipFile,
    entries: List[str],
    recorded: List[str],
    session: Session,
) -> List[str]:
    """Copy the archive's frames into the session, keeping their names.

    The playback order comes from the archive's recorded `file_names` when they
    account for every entry, because a name like `frame_2.jpg` cannot be ordered
    reliably on its own; otherwise the entry names are sorted naturally.
    """
    by_name = {
        os.path.basename(entry): entry
        for entry in entries
        if os.path.splitext(entry)[1].lower() in IMAGE_EXTENSIONS
    }
    if not by_name:
        raise InvalidRequest(
            "The archive's frames/ folder holds no images "
            f"({', '.join(IMAGE_EXTENSIONS)})."
        )

    if recorded and len(recorded) == len(by_name) and set(recorded) == set(by_name):
        ordered = list(recorded)
    else:
        ordered = sorted(by_name, key=natural_key)

    os.makedirs(session.frames_dir, exist_ok=True)
    for name in ordered:
        destination = os.path.join(session.frames_dir, name)
        with archive.open(by_name[name]) as source, open(destination, "wb") as sink:
            shutil.copyfileobj(source, sink)
    return ordered


def _extract_video(
    archive: zipfile.ZipFile,
    entry: str,
    session: Session,
    settings: Settings,
    *,
    fps: float,
) -> List[str]:
    """Write the embedded video out, then decode it into the session's frames."""
    os.makedirs(session.source_dir, exist_ok=True)
    name = sanitize_name(os.path.basename(entry), fallback="source.mp4")
    video_path = os.path.join(session.source_dir, name)
    with archive.open(entry) as source, open(video_path, "wb") as sink:
        shutil.copyfileobj(source, sink)

    return extract_frames(
        video_path,
        session.frames_dir,
        fps=fps,
        jpeg_quality=settings.frames_jpeg_quality,
        ffmpeg=settings.ffmpeg_bin,
    )


def load_archive_into_session(
    archive_path: str,
    session: Session,
    settings: Settings,
) -> LoadedProject:
    """Fill `session.frames_dir` from the archive at `archive_path`.

    Raises `UnsupportedMediaType` when the upload is not a ZIP and
    `InvalidRequest` when it carries neither frames nor a video.
    """
    if not zipfile.is_zipfile(archive_path):
        raise UnsupportedMediaType(
            "The uploaded project is not a ZIP archive (expected a `.project`)."
        )

    with zipfile.ZipFile(archive_path) as archive:
        dataset, annotation_entry = _read_dataset(archive)
        videos = dataset.get("videos")
        record = videos[0] if isinstance(videos, list) and videos else {}
        if not isinstance(record, dict):
            record = {}

        recorded = [
            str(name)
            for name in (record.get("file_names") or [])
            if isinstance(name, str) and name
        ]
        recorded_fps = _number(record.get("fps"))
        requested_fps = _number(record.get("target_fps"))
        # The rate an extraction should aim for: what the archive asked for.
        extract_fps = requested_fps or recorded_fps or DEFAULT_FPS_FALLBACK
        mode = record.get("segmentation_mode")
        mode = str(mode) if isinstance(mode, str) and mode else None
        original_fps = _number(record.get("original_fps"))

        frame_entries = _entry_names(archive, FRAMES_DIR)
        video_entries = _entry_names(archive, VIDEO_DIR)

        if frame_entries:
            frame_names = _copy_frames(archive, frame_entries, recorded, session)
            source = "frames"
            video_entry = None
            fps = recorded_fps or extract_fps
        elif video_entries:
            video_entry = sorted(video_entries)[0]
            frame_names = _extract_video(
                archive, video_entry, session, settings, fps=extract_fps
            )
            source = "video"
            fps = extract_fps
            if original_fps is None:
                # Provenance only, so a container ffprobe cannot read is fine.
                try:
                    original_fps = probe_video(
                        os.path.join(
                            session.source_dir,
                            sanitize_name(
                                os.path.basename(video_entry), fallback="source.mp4"
                            ),
                        ),
                        ffprobe=settings.ffprobe_bin,
                    ).fps
                except Exception:  # noqa: BLE001 - purely informational
                    original_fps = None
        else:
            raise InvalidRequest(
                "The archive holds neither a frames/ folder nor a video/ file, "
                "so there is nothing to review."
            )

    width, height = frame_dimensions(session.frames_dir)
    loaded = LoadedProject(
        source=source,
        frame_names=frame_names,
        fps=fps,
        width=width,
        height=height,
        mode=mode,
        original_fps=original_fps,
        video_entry=video_entry,
        archive_name=os.path.basename(archive_path),
        annotation_entry=annotation_entry,
        recorded_frame_names=recorded or None,
    )
    session.update_meta(
        source=loaded.source,
        frame_names=loaded.frame_names,
        frame_count=loaded.frame_count,
        fps=loaded.fps,
        width=loaded.width,
        height=loaded.height,
        mode=loaded.mode,
        original_fps=loaded.original_fps,
        video_entry=loaded.video_entry,
        annotation_entry=loaded.annotation_entry,
        recorded_frame_names=loaded.recorded_frame_names or [],
        frame_names_match=loaded.frame_names_match,
    )
    return loaded

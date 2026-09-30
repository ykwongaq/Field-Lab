"""The `.project` archive layout, and the pure name helpers that go with it.

This is the *contract*, not an implementation: what a `.project` archive is made
of — the directory names, the annotation and metadata entries, the accepted image
and video extensions, how frames are named inside one — and nothing about how an
archive is written.

It lives apart from `builder` for that reason. The reader (`loader`) needs the
contract and nothing else, but it used to import the writer to get it, and so
pulled in a module that owns video encoding, resampling and the OpenCV image
stack — none of which a reader has any use for, and one of which is an optional
dependency. Both sides now depend on this module and neither on the other.

Nothing here imports cv2, torch, or anything from `domain`/`inference`.
"""

from __future__ import annotations

import os
import re
from typing import Any, List

from src.core.errors import InvalidRequest

FRAMES_DIR = "frames"
VIDEO_DIR = "video"
ANNOTATION_ENTRY = "annotation.json"
#: A project archive is a ZIP named `.project`, plus the entry that holds the
#: caller's free-form metadata.
METADATA_ENTRY = "metadata.json"
PROJECT_EXTENSION = ".project"

#: How frames are named *inside an archive*. Deliberately not the 8-digit pattern
#: `core.sessions` uses for a session's frame directory: those are two different
#: contracts, so they get two different names.
ARCHIVE_FRAME_PATTERN = "{:06d}.jpg"

#: Images accepted when the source is a folder of frames rather than a video.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

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

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_DIGITS = re.compile(r"(\d+)")


def archive_frame_name(index: int) -> str:
    """Name of frame `index` inside a project archive."""
    return ARCHIVE_FRAME_PATTERN.format(index)


def sanitize_name(
    raw: str, *, from_filename: bool = False, fallback: str = "project"
) -> str:
    """A name that is safe to use as a path component and inside a ZIP."""
    text = raw or ""
    if from_filename:
        text = os.path.splitext(os.path.basename(text.replace("\\", "/")))[0]
    cleaned = _SAFE_NAME.sub("_", text).strip("._-")
    return cleaned or fallback


def project_name_for(name: str | None, filename: str) -> str:
    """The project's name: the caller's, or the source filename's."""
    if name and name.strip():
        return sanitize_name(name)
    return sanitize_name(filename, from_filename=True)


def natural_key(name: str) -> List[Any]:
    """Sort key that orders `frame_2.jpg` before `frame_10.jpg`."""
    return [
        int(part) if part.isdigit() else part.lower() for part in _DIGITS.split(name)
    ]


def select_frame_files(folder: str) -> List[str]:
    """The image files of a frame folder, in natural name order."""
    if not os.path.isdir(folder):
        raise InvalidRequest(f"Frame folder not found: {folder!r}")
    names = [
        name
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
        and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not names:
        raise InvalidRequest(
            f"Frame folder {folder!r} holds no images "
            f"({', '.join(IMAGE_EXTENSIONS)})."
        )
    names.sort(key=natural_key)
    return names

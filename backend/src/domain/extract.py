"""Turn a video into the one frame format every reader works from.

A session's ``frames/`` directory is the single frame source for the annotation
panel, SAM 3 and propagation. Whatever the archive carried, the video is decoded
to plain JPEGs numbered from ``00000000.jpg`` up, so no reader has to know where
the pixels came from.

ffmpeg's image2 muxer picks its own start index, and that has shifted between
builds, so frames are extracted into a scratch directory and renumbered into
``frames/`` afterwards. That keeps the archive's ``file_names`` contract exact
rather than assumed, and is why nothing here tries to read the numbering back
out of ffmpeg's own output.

Two ffmpeg behaviours are relied on deliberately: it autorotates by default
(so a portrait clip's frames match what the browser reported), and ``fps=``
resamples to the target rate exactly rather than decimating by an integer step.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from src.core.errors import InvalidRequest, Unavailable
from src.core.sessions import frame_name, list_frame_names

DEFAULT_FFMPEG = "ffmpeg"
DEFAULT_FFPROBE = "ffprobe"

#: Scratch directory a session extracts into before the frames are renumbered.
SCRATCH_DIRNAME = "frames.raw"

#: ffmpeg's ``-q:v`` scale is inverted — a lower number is better quality.
#: 2 is roughly 95%, 3 roughly 92%, 5 roughly 80%.
_JPEG_QUALITY_TO_QSCALE = {
    100: 1,
    95: 2,
    90: 3,
    85: 4,
    80: 5,
}
_DEFAULT_QSCALE = 2  # 95%

EXTRACT_TIMEOUT_SECONDS = 60 * 60
PROBE_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class VideoInfo:
    """What ffprobe could tell us about a source video.

    Every field is optional on purpose: containers lie or omit, and none of this
    is authoritative once the frames exist — ``frame_dimensions`` measures the
    decoded pixels instead.
    """

    fps: Optional[float]
    width: int
    height: int
    duration_seconds: Optional[float]
    frame_count: Optional[int]


def qscale_for(quality: int) -> int:
    """Map a 1..100 JPEG quality onto ffmpeg's inverted ``-q:v`` scale."""
    if not 1 <= quality <= 100:
        raise InvalidRequest("JPEG quality must be within 1..100.")
    if quality in _JPEG_QUALITY_TO_QSCALE:
        return _JPEG_QUALITY_TO_QSCALE[quality]
    # Linear fit on the anchors above, clamped to the usable range.
    return max(1, min(31, round(1 + (100 - quality) / 5)))


def missing_binaries(
    ffmpeg: str = DEFAULT_FFMPEG, ffprobe: str = DEFAULT_FFPROBE
) -> List[str]:
    """Which of ffmpeg/ffprobe cannot be resolved; empty when both are usable.

    Checked once at startup so a machine without them is reported on the way up
    rather than on the first upload, where the failure reads as a broken project.
    A process only sees the PATH it was launched with, so installing ffmpeg into
    a running service's parent shell is not enough — it has to be restarted.
    """
    return [name for name in (ffmpeg, ffprobe) if shutil.which(name) is None]


def _ratio(value: object) -> Optional[float]:
    """Parse ffprobe's ``"25/1"`` style ratio."""
    if not isinstance(value, str) or not value:
        return None
    if "/" in value:
        numerator, _, denominator = value.partition("/")
        try:
            top, bottom = float(numerator), float(denominator)
        except ValueError:
            return None
        return top / bottom if bottom else None
    try:
        return float(value)
    except ValueError:
        return None


def _number(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _run(command: List[str], *, timeout: int, label: str) -> str:
    """Run a subprocess, turning its failure modes into backend errors."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise Unavailable(
            f"{label} is not installed or not on PATH ({command[0]!r}). "
            "Install it, or point VSR_FFMPEG_BIN / VSR_FFPROBE_BIN at the binary."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise InvalidRequest(f"{label} timed out after {timeout} seconds.") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {completed.returncode}"
        raise InvalidRequest(f"{label} failed: {tail}")
    return completed.stdout


def probe_video(path: str, *, ffprobe: str = DEFAULT_FFPROBE) -> VideoInfo:
    """Read a video's source frame rate, size and duration.

    Used for provenance only — the archive records what the source was, never
    what the frames are, because those are measured after extraction.
    """
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        path,
    ]
    raw = _run(command, timeout=PROBE_TIMEOUT_SECONDS, label="ffprobe")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise InvalidRequest(f"ffprobe returned no usable JSON: {exc}") from exc

    streams = payload.get("streams") or [{}]
    stream = streams[0] if isinstance(streams[0], dict) else {}
    duration = _number(stream.get("duration"))
    if duration is None:
        duration = _number((payload.get("format") or {}).get("duration"))
    frames = _number(stream.get("nb_frames"))

    return VideoInfo(
        fps=_ratio(stream.get("r_frame_rate")) or _ratio(stream.get("avg_frame_rate")),
        width=int(_number(stream.get("width")) or 0),
        height=int(_number(stream.get("height")) or 0),
        duration_seconds=duration,
        frame_count=int(frames) if frames is not None else None,
    )


def extract_frames(
    video_path: str,
    out_dir: str,
    *,
    fps: float,
    jpeg_quality: int = 95,
    ffmpeg: str = DEFAULT_FFMPEG,
    scratch_dir: Optional[str] = None,
    timeout: int = EXTRACT_TIMEOUT_SECONDS,
) -> List[str]:
    """Write ``video_path`` to ``out_dir`` as ``00000000.jpg`` and up.

    Returns the frame names written, in order. The frames land in a scratch
    directory first and are renumbered on the way out, so the caller gets a
    contiguous sequence starting at zero whatever ffmpeg chose to emit.
    """
    if fps <= 0:
        raise InvalidRequest("The target frame rate must be above zero.")
    if not os.path.isfile(video_path):
        raise InvalidRequest(f"Video not found: {video_path!r}")

    raw_dir = scratch_dir or os.path.join(os.path.dirname(out_dir), SCRATCH_DIRNAME)
    shutil.rmtree(raw_dir, ignore_errors=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(qscale_for(jpeg_quality)),
        "-start_number",
        "0",
        "-f",
        "image2",
        os.path.join(raw_dir, "%08d.jpg"),
    ]

    try:
        _run(command, timeout=timeout, label="ffmpeg")
        produced = list_frame_names(raw_dir)
        if not produced:
            raise InvalidRequest(
                "No frames could be decoded from the video "
                "(unsupported codec, or an empty file)."
            )
        return _renumber(raw_dir, out_dir, produced)
    finally:
        # The raw directory is scratch; the renumbered frames are the result.
        shutil.rmtree(raw_dir, ignore_errors=True)


def _renumber(raw_dir: str, out_dir: str, produced: List[str]) -> List[str]:
    """Move the extracted frames into ``out_dir`` as a zero-based sequence."""
    written: List[str] = []
    for index, name in enumerate(produced):
        target = frame_name(index)
        os.replace(os.path.join(raw_dir, name), os.path.join(out_dir, target))
        written.append(target)
    return written


def frame_dimensions(frames_dir: str) -> tuple:
    """``(width, height)`` of the first frame, read from its header.

    Measured rather than taken from :func:`probe_video`, because ffprobe reports
    the stored stream size while ffmpeg autorotates — a portrait clip would
    otherwise describe itself transposed. Only the header is parsed, so this
    stays cheap even on a 4K frame.
    """
    names = list_frame_names(frames_dir)
    if not names:
        raise InvalidRequest("The session holds no frames yet.")
    from PIL import Image  # local, so the frame maths never needs Pillow

    with Image.open(os.path.join(frames_dir, names[0])) as image:
        width, height = image.size
    return width, height

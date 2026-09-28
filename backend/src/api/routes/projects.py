"""Build a reviewer project archive from an uploaded video."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from src.api.deps import SettingsDep, stream_upload_to_path
from src.core.errors import InvalidRequest, UnsupportedMediaType
from src.projects.builder import (
    MODES,
    VIDEO_EXTENSIONS,
    build_project_from_video,
    project_name_for,
)

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _remove_tree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


@router.post("/create")
async def create_project(
    background: BackgroundTasks,
    settings: SettingsDep,
    video: UploadFile = File(..., description="Source video file"),
    mode: str = Form(..., description="'instance' or 'semantic'"),
    name: Optional[str] = Form(None, description="Project name"),
    frame_step: int = Form(1, ge=1, description="Keep every N-th frame"),
    jpeg_quality: int = Form(90, ge=1, le=100),
) -> FileResponse:
    """Build a project ZIP from an uploaded video and stream it back.

    The response is the archive itself (`application/zip`) with a
    `Content-Disposition` filename plus `X-Project-*` summary headers, so the
    browser can both save the file and open it immediately in the reviewer.
    """
    if mode not in MODES:
        raise InvalidRequest(f"`mode` must be one of {', '.join(MODES)}")

    original_name = video.filename or "upload.mp4"
    ext = os.path.splitext(original_name)[1].lower()
    if ext and ext not in VIDEO_EXTENSIONS:
        raise UnsupportedMediaType(
            f"Unsupported video extension {ext!r}; expected one of "
            + ", ".join(VIDEO_EXTENSIONS)
        )

    # The archive is built on disk and cleaned up once the response is sent.
    workdir = tempfile.mkdtemp(prefix="vsr-project-")
    background.add_task(_remove_tree, workdir)

    video_path = os.path.join(workdir, "source" + (ext or ".mp4"))
    await stream_upload_to_path(
        video, video_path, settings.max_upload_bytes, label="Video"
    )

    project_name = project_name_for(name, original_name)
    output_zip = os.path.join(workdir, project_name + ".zip")
    result = await run_in_threadpool(
        build_project_from_video,
        video_path,
        output_zip,
        mode=mode,
        name=project_name,
        frame_step=frame_step,
        jpeg_quality=jpeg_quality,
        source_filename=original_name,
    )

    return FileResponse(
        output_zip,
        media_type="application/zip",
        filename=f"{result.name}.zip",
        headers={
            "X-Project-Name": result.name,
            "X-Project-Mode": result.mode,
            "X-Project-Frames": str(result.frame_count),
            "X-Project-Fps": f"{result.fps:.6g}",
        },
    )

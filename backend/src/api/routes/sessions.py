"""Open a project archive into a session, and serve its frames.

A session is the one place frames live for the duration of a review: an archive
carrying a frame folder has it copied in, one carrying a video has its frames
decoded in by ffmpeg. Every reader — the annotation panel, SAM 3, propagation —
then works from the same numbered sequence, so none of them needs to know what
the archive held.
"""

from __future__ import annotations

import mimetypes
import os

from fastapi import APIRouter, File, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from src.api.deps import SettingsDep, stream_upload_to_path
from src.api.serializers import session_response
from src.core.errors import NotFound
from src.core.sessions import (
    create_session,
    delete_session,
    open_session,
)
from src.projects.loader import load_archive_into_session
from src.schemas.sessions import SessionResponse

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

#: The uploaded archive is kept under this fixed name, so it can never collide
#: with the video an archive may also carry into the same directory.
ARCHIVE_NAME = "archive.project"


@router.post("", response_model=SessionResponse, status_code=201)
async def open_project(
    settings: SettingsDep,
    project: UploadFile = File(
        ..., description="The `.project` archive to open (a ZIP)"
    ),
) -> SessionResponse:
    """Take an archive in, materialise its frames, and hand back the session.

    This waits for the frames: a frame folder is copied, a video is decoded, and
    the caller needs the frame list before it can show anything. A session that
    fails to build is deleted rather than left half-populated on disk.
    """
    session = create_session(settings.temp_dir, meta={"archive": project.filename})
    try:
        archive_path = os.path.join(session.source_dir, ARCHIVE_NAME)
        await stream_upload_to_path(
            project,
            archive_path,
            settings.max_upload_bytes,
            label="Project archive",
        )
        await run_in_threadpool(
            load_archive_into_session, archive_path, session, settings
        )
    except Exception:
        session.delete()
        raise
    session.touch()
    return session_response(session, settings)


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, settings: SettingsDep) -> SessionResponse:
    """Describe a session again, so a reloaded page can resume the review."""
    session = open_session(settings.temp_dir, session_id)
    session.touch()
    return session_response(session, settings)


@router.get("/{session_id}/frames/{index}")
def get_frame(session_id: str, index: int, settings: SettingsDep) -> FileResponse:
    """One frame, exactly as the session stored it.

    A session's frames never change once it is built, so they are marked
    immutable and the browser can cache them for the life of the session.
    """
    session = open_session(settings.temp_dir, session_id)
    # `basename` so a recorded name can never reach outside `frames/`.
    name = os.path.basename(session.frame_name_at(index))
    path = os.path.join(session.frames_dir, name)
    if not os.path.isfile(path):
        raise NotFound(f"Frame {index} is not part of session {session_id}.")
    session.touch()
    return FileResponse(
        path,
        media_type=mimetypes.guess_type(path)[0] or "image/jpeg",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.delete("/{session_id}", status_code=204)
def close_session(session_id: str, settings: SettingsDep) -> Response:
    """Drop a session and its frames.

    Idempotent on purpose: the sweeper may already have taken an idle session,
    and the usual caller is a page-unload beacon, which cannot retry or report.
    """
    delete_session(settings.temp_dir, session_id)
    return Response(status_code=204)

"""Session DTOs: what an opened project looks like over HTTP."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class SessionResponse(BaseModel):
    """A materialised project: the frame sequence a session now holds."""

    session_id: str
    #: `"frames"` when the archive carried them, `"video"` when they were decoded.
    source: str
    frame_names: List[str]
    frame_count: int
    fps: float
    width: int
    height: int
    #: Segmentation mode the archive declared, if it declared one.
    mode: Optional[str] = None
    original_fps: Optional[float] = None
    video_entry: Optional[str] = None
    #: The archive this session was built from, as uploaded.
    archive: Optional[str] = None
    #: What the archive's own `file_names` claimed, for drift detection.
    recorded_frame_names: List[str] = []
    frame_names_match: bool = True
    #: How long the session survives without being used again.
    expires_in_seconds: int = 0

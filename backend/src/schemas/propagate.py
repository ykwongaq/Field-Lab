"""Propagation job DTOs.

Propagation is asynchronous: a run can cover thousands of frames, so the caller
starts a job and then polls it. `since` on a poll returns only the frames
produced after that index, which keeps the payload proportional to what is new
rather than to the whole run — that is what lets the reviewer paint masks as they
arrive without re-downloading the ones it already has.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.schemas.rle import RleMask
from src.schemas.sam import ModelStatus


class PropagateStatus(BaseModel):
    """Tracker availability, the window sizing in force, and how busy the GPU is."""

    sam3: ModelStatus
    window_frames: int
    overlap: int
    anchor_max: int
    chaining: str = "derived"
    #: Jobs waiting for the worker and jobs running (0 or 1). Global rather than
    #: per caller: the queue belongs to the process, and it is the one GPU
    #: everybody shares.
    queued_jobs: int = 0
    running_jobs: int = 0
    #: A model call in flight right now, and how many callers are queued for it.
    #: Process-wide, so it covers clicks as well as propagation runs.
    gpu_busy: bool = False
    gpu_waiting: int = 0


class PinnedMask(BaseModel):
    """One frame a human verified by hand, carried into a refinement run.

    A pin is written into tracker memory as authoritative, exactly like the
    anchor, so it is the *only* kind of frame that may re-anchor a window: model
    output must never be promoted to conditioning data. The mask travels in the
    body for the same reason the anchor's does — the annotation lives in the
    browser's archive, while the frames are already in the session.
    """

    frame_index: int = Field(..., ge=0)
    mask: RleMask


class PropagationRequest(BaseModel):
    """Start a propagation job for one mask.

    `first`/`last` are inclusive frame indices; leaving them out covers the whole
    clip, which is the default the reviewer offers. The mask travels in the body
    because the annotation lives in the browser's archive, while the frames do
    not: they are already in the session.

    `pins` are the frames the reviewer corrected by hand. A run seeded with them
    is a *refinement*: it restarts from verified masks instead of from its own
    previous output.
    """

    session_id: str
    anchor_frame: int = Field(..., ge=0)
    mask: RleMask
    direction: str = Field("both", description="forward, backward or both")
    first: Optional[int] = Field(None, ge=0)
    last: Optional[int] = Field(None, ge=0)
    object_id: Optional[int] = None
    pins: List[PinnedMask] = Field(default_factory=list)
    chaining: Optional[Literal["derived", "verified"]] = Field(
        None,
        description=(
            "derived (default): a window with no verified frame is seeded from the "
            "previous window's output. verified: the run stops instead."
        ),
    )


class JobProgressResponse(BaseModel):
    """How far along a run is, in frames and windows."""

    frames_done: int = 0
    frames_total: int = 0
    window_index: int = 0
    windows_total: int = 0


class PropagatedFrameResponse(BaseModel):
    """One frame's mask. Decoded runs are omitted: the client decodes RLE."""

    frame_index: int
    rle: RleMask
    area: int


class JobResponse(BaseModel):
    """A job's state, its plan, and (when asked) the masks produced so far."""

    job_id: str
    session_id: str
    object_id: Optional[int] = None
    state: str
    error: Optional[str] = None
    anchor: int
    direction: str
    first: int
    last: int
    progress: JobProgressResponse
    plan: Optional[Dict[str, Any]] = None
    #: 0 while the job is running; how many jobs are ahead of it otherwise.
    queue_position: int = 0
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    updated_at: float
    masks: List[PropagatedFrameResponse] = Field(default_factory=list)


class JobListResponse(BaseModel):
    """The queue, oldest first, so the reviewer can show what is waiting."""

    jobs: List[JobResponse] = Field(default_factory=list)
    running: Optional[str] = None

"""Propagating one mask across a clip, as a background job.

A run can cover every frame of a long clip, so it cannot be a single request:
the reviewer starts a job, watches masks arrive, and can cancel. The frames come
from the session (the clip is already open), while the anchor mask travels in the
body, because the annotation belongs to the browser's archive rather than to the
backend.

One job propagates one mask. Asking for several objects means several jobs, which
the queue runs back to back — that is what bounds GPU memory to one object's
tracker state instead of growing with the number of objects.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from fastapi import APIRouter

from src.api.deps import ClientDep, JobsDep, PropagateDep, SettingsDep, require_sam3
from src.api.serializers import job_is_active, job_response
from src.core.errors import InvalidRequest
from src.core.sessions import open_session
from src.domain.rle import decode_rle
from src.inference.frames import frame_size
from src.inference.sam3_video import PropagateError
from src.schemas.propagate import (
    JobListResponse,
    JobResponse,
    PinnedMask,
    PropagationRequest,
    PropagateStatus,
)
from src.schemas.sam import ModelStatus

router = APIRouter(prefix="/api/propagate", tags=["propagate"])


def _decode_pins(
    pins: Sequence[PinnedMask], *, anchor: int, width: int, height: int
) -> Dict[int, np.ndarray]:
    """Decode and check the frames the reviewer verified by hand.

    A pin is written into tracker memory as authoritative, exactly like the
    anchor, so it gets the same scrutiny: the right shape, a real object in it,
    and no duplicate of a frame that is already spoken for. An empty pin is
    rejected rather than ignored because "the object is not on this frame" is a
    *cleared* frame (no mask at all), not a mask full of zeros.
    """
    decoded: Dict[int, np.ndarray] = {}
    for pin in pins:
        if pin.frame_index == anchor:
            raise InvalidRequest(
                f"Verified frame {pin.frame_index} is the anchor frame; its mask "
                "travels as `mask`, not as a pin."
            )
        if pin.frame_index in decoded:
            raise InvalidRequest(
                f"Verified frame {pin.frame_index} was sent more than once."
            )
        mask = decode_rle(pin.mask.size, pin.mask.counts)
        if mask.shape != (height, width):
            raise PropagateError(
                f"The mask for verified frame {pin.frame_index} is "
                f"{mask.shape[1]}x{mask.shape[0]} but the frames are "
                f"{width}x{height}."
            )
        if not mask.any():
            raise InvalidRequest(
                f"The mask for verified frame {pin.frame_index} is empty. A frame "
                "with no object is a cleared frame, not a pin."
            )
        decoded[pin.frame_index] = mask
    return decoded


@router.get("/status", response_model=PropagateStatus)
def propagate_status(
    settings: SettingsDep, service: PropagateDep, jobs: JobsDep
) -> PropagateStatus:
    """Report tracker availability, the window sizing, and how busy the GPU is.

    The queue and gate numbers are global rather than scoped to the caller: they
    describe the one GPU everybody shares, which is what explains why a run that
    was just accepted has not started moving yet. None of this is user data, so
    the endpoint stays open like `/api/sam3/status`.
    """
    service_status = service.status()
    sam3 = ModelStatus(
        available=bool(service_status["available"]),
        loaded=bool(service_status["loaded"]),
        model=str(service_status["model"]),
        device=str(service_status["device"]),
        error=service_status.get("error"),
    )
    counts = jobs.counts()
    return PropagateStatus(
        sam3=sam3,
        window_frames=int(service_status["window_frames"]),
        overlap=int(service_status["overlap"]),
        anchor_max=int(service_status["anchor_max"]),
        chaining=str(service_status.get("chaining", "derived")),
        queued_jobs=int(counts["queued"]),
        running_jobs=int(counts["running"]),
        gpu_busy=bool(service_status.get("gpu_busy", False)),
        gpu_waiting=int(service_status.get("gpu_waiting", 0)),
    )


@router.post("/jobs", response_model=JobResponse, status_code=202)
def start_job(
    request: PropagationRequest,
    settings: SettingsDep,
    service: PropagateDep,
    jobs: JobsDep,
    client_id: ClientDep,
) -> JobResponse:
    """Queue a propagation run and return it straight away.

    The whole request is validated here — the mask against the frame size, the
    range against the clip, and the range against the anchor — so a malformed run
    is a 422 rather than a job that fails a minute later.
    """
    require_sam3(settings)
    session = open_session(settings.temp_dir, request.session_id, owner=client_id)
    frame_count = session.frame_count()
    if frame_count == 0:
        raise InvalidRequest(
            f"Session {session.id} has no frames, so there is nothing to propagate."
        )
    if request.anchor_frame >= frame_count:
        raise InvalidRequest(
            f"Frame {request.anchor_frame} is past the end of the clip "
            f"({frame_count} frames)."
        )

    width, height = frame_size(session, request.anchor_frame)
    mask = decode_rle(request.mask.size, request.mask.counts)
    if mask.shape != (height, width):
        raise PropagateError(
            f"The mask is {mask.shape[1]}x{mask.shape[0]} but the frames are "
            f"{width}x{height}."
        )
    if not mask.any():
        raise InvalidRequest(
            "The anchor mask is empty, so there is no object to propagate. Draw or "
            "pick a mask on the anchor frame first."
        )
    pins = _decode_pins(
        request.pins, anchor=request.anchor_frame, width=width, height=height
    )

    first = 0 if request.first is None else request.first
    last = (
        frame_count - 1 if request.last is None else min(request.last, frame_count - 1)
    )
    plan = service.plan(
        anchor=request.anchor_frame,
        first=first,
        last=last,
        direction=request.direction,
        frame_count=frame_count,
    )

    # A pin the plan never visits could not be injected (SAM 3 addresses frames
    # inside one session), so it is a mistake in the request rather than a pin
    # that quietly does nothing.
    outside = sorted(frame for frame in pins if not plan.covers(frame))
    if outside:
        raise InvalidRequest(
            f"Verified frame(s) {outside} are outside the frames this run covers "
            f"({plan.first}..{plan.last}, {plan.direction})."
        )

    position = jobs.queued_count()
    if plan.frames_total <= 1:
        raise InvalidRequest(
            f"Frames {plan.first}..{plan.last} hold only the anchor frame, so there "
            "is nothing to propagate. Widen the range."
        )
    job = jobs.submit(
        session_id=session.id,
        client_id=client_id,
        anchor=request.anchor_frame,
        direction=plan.direction,
        first=plan.first,
        last=plan.last,
        anchor_mask=mask,
        frame_count=frame_count,
        object_id=request.object_id,
        pins=pins,
        chaining=request.chaining,
    )
    session.touch()
    return job_response(job, include_masks=False, queue_position=position)


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(jobs: JobsDep, client_id: ClientDep) -> JobListResponse:
    """This client's queue, oldest first, so the reviewer can show what is waiting.

    Scoped to the caller on purpose: the queue itself is global (one GPU runs one
    job at a time), but a client sees only its own runs, so another reviewer's
    work is neither listed nor countable.
    """
    known = jobs.list(client_id=client_id)
    running = next((job.id for job in known if job_is_active(job)), None)
    return JobListResponse(
        jobs=[job_response(job, include_masks=False) for job in known],
        running=running,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    jobs: JobsDep,
    client_id: ClientDep,
    since: int = -1,
    include_masks: bool = True,
) -> JobResponse:
    """Poll one job, optionally fetching only the masks newer than `since`.

    `since` is the highest frame index the caller already holds, which is what
    keeps a poll's payload proportional to the new frames rather than to the
    whole run.
    """
    job = jobs.get(job_id, client_id=client_id)
    return job_response(job, since=since, include_masks=include_masks)


@router.delete("/jobs/{job_id}", response_model=JobResponse)
def cancel_job(job_id: str, jobs: JobsDep, client_id: ClientDep) -> JobResponse:
    """Ask a job to stop; a queued job never starts, a running one stops soon.

    Whatever it has already produced is kept, so the caller can still accept the
    frames that were computed before the cancel.
    """
    job = jobs.cancel(job_id, client_id=client_id)
    return job_response(job, include_masks=False)

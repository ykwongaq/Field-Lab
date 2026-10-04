"""Domain and inference results -> response DTOs.

Keeping the translation here means `schemas/` stays pure data and the routers
stay free of numpy and pycocotools.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from src.core.config import Settings
from src.core.jobs import JobState, PropagationJob
from src.core.sessions import Session
from src.domain.prompts import InstanceMask, SegmentResult
from src.domain.rle import Run, binary_to_runs, encode_rle
from src.schemas.propagate import (
    JobProgressResponse,
    JobResponse,
    PropagatedFrameResponse,
)
from src.schemas.rle import ForegroundRun, RleMask
from src.schemas.sam import InstanceResponse, SegmentResponse
from src.schemas.sessions import SessionResponse


def session_response(session: Session, settings: Settings) -> SessionResponse:
    """Describe a materialised session from its provenance file.

    Everything the response needs was written when the session was built, so
    reconnecting after a page reload is a read rather than a re-extraction.
    """
    meta = session.read_meta()
    recorded = meta.get("recorded_frame_names")
    if not isinstance(recorded, list):
        recorded = []
    return SessionResponse(
        session_id=session.id,
        source=str(meta.get("source") or "frames"),
        frame_names=session.frame_names(),
        frame_count=session.frame_count(),
        fps=float(meta.get("fps") or 0.0),
        width=int(meta.get("width") or 0),
        height=int(meta.get("height") or 0),
        mode=meta.get("mode"),
        original_fps=meta.get("original_fps"),
        video_entry=meta.get("video_entry"),
        archive=meta.get("archive"),
        recorded_frame_names=[str(name) for name in recorded],
        frame_names_match=bool(meta.get("frame_names_match", True)),
        expires_in_seconds=int(settings.session_ttl_seconds),
    )


def to_foreground_runs(runs: Iterable[Run]) -> List[ForegroundRun]:
    """Adapt `domain.rle` run tuples into the response model."""
    return [ForegroundRun(x=x, y=y, length=length) for x, y, length in runs]


def rle_mask(mask: np.ndarray) -> RleMask:
    """Encode a bool (H, W) mask as a response-ready `RleMask`."""
    size, counts = encode_rle(mask)
    return RleMask(size=size, counts=counts)


def _mask_runs(mask: np.ndarray) -> List[ForegroundRun]:
    return to_foreground_runs(binary_to_runs(mask.astype(np.uint8)))


def instance_response(instance: InstanceMask) -> InstanceResponse:
    """One proposed mask, with its score and extent."""
    bbox = instance.bbox()
    return InstanceResponse(
        rle=rle_mask(instance.mask),
        runs=_mask_runs(instance.mask),
        score=round(float(instance.score), 4),
        area=instance.area,
        bbox=list(bbox) if bbox else None,
    )


def segment_response(
    result: SegmentResult, *, max_instances: int = 0
) -> SegmentResponse:
    """Build the segmentation response: the mask to use plus its alternatives.

    `max_instances` trims the alternatives, which matters for a text prompt on a
    crowded frame: the caller can ask for just the best few instead of every
    match.
    """
    instances = list(result.instances)
    if max_instances > 0:
        instances = sorted(instances, key=lambda item: item.score, reverse=True)[
            :max_instances
        ]
    bbox = result.bbox()
    return SegmentResponse(
        rle=rle_mask(result.mask),
        runs=_mask_runs(result.mask),
        height=result.height,
        width=result.width,
        score=round(float(result.score), 4),
        area=result.area,
        bbox=list(bbox) if bbox else None,
        kind=result.kind,
        prompt=result.prompt,
        embedding_reused=result.embedding_reused,
        encoder_ms=round(result.encoder_ms, 1),
        decoder_ms=round(result.decoder_ms, 1),
        instances=[instance_response(item) for item in instances],
        instance_scores=[round(float(item.score), 4) for item in instances],
    )


def _propagated_frame(frame_index: int, mask: np.ndarray) -> PropagatedFrameResponse:
    """One produced frame. Runs are omitted: the client decodes the RLE itself."""
    return PropagatedFrameResponse(
        frame_index=frame_index,
        rle=rle_mask(mask),
        area=int(mask.sum()),
    )


def job_response(
    job: PropagationJob,
    *,
    since: Optional[int] = None,
    until: Optional[int] = None,
    include_masks: bool = True,
    queue_position: int = 0,
) -> JobResponse:
    """Describe a propagation job, optionally with the masks it has produced.

    `since`/`until` keep a poll cheap: the caller passes the highest and lowest
    frame it already has and receives only what lies outside that span, so a run
    that propagates outward from its anchor does not re-send — or drop — either
    end of the clip.
    """
    masks: Dict[int, np.ndarray] = {}
    progress = JobProgressResponse(**job.progress.as_dict())
    if include_masks:
        masks, snapshot = job.snapshot(since=since, until=until)
        progress = JobProgressResponse(**snapshot.as_dict())

    summary = job.summary()
    return JobResponse(
        job_id=summary["job_id"],
        session_id=summary["session_id"],
        object_id=summary["object_id"],
        state=summary["state"],
        error=summary["error"],
        anchor=summary["anchor"],
        direction=summary["direction"],
        first=summary["first"],
        last=summary["last"],
        progress=progress,
        plan=summary["plan"],
        queue_position=queue_position,
        created_at=summary["created_at"],
        started_at=summary["started_at"],
        finished_at=summary["finished_at"],
        updated_at=summary["updated_at"],
        masks=[
            _propagated_frame(frame_index, masks[frame_index])
            for frame_index in sorted(masks)
        ],
    )


def job_is_active(job: PropagationJob) -> bool:
    """Whether a job still occupies the queue."""
    return job.state in (JobState.QUEUED, JobState.RUNNING)

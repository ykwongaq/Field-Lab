"""Domain and inference results -> response DTOs.

Keeping the translation here means `schemas/` stays pure data and the routers
stay free of numpy and pycocotools.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from src.core.config import Settings
from src.core.sessions import Session
from src.domain.rle import Run, binary_to_runs, encode_rle
from src.domain.segmentation import ConceptResult, SegmentResult
from src.inference.propagate import PropagateResult
from src.schemas.masks import ForegroundRun, RleMask
from src.schemas.propagate import PropagateResponse, PropagatedFrame
from src.schemas.sam import Sam3SegmentResponse, SegmentResponse
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


def segment_response(result: SegmentResult) -> SegmentResponse:
    """Build the mask response shared by every segmentation endpoint."""
    bbox = result.bbox()
    return SegmentResponse(
        rle=rle_mask(result.mask),
        runs=_mask_runs(result.mask),
        height=result.height,
        width=result.width,
        score=result.score,
        area=result.area,
        bbox=list(bbox) if bbox else None,
        embedding_reused=result.embedding_reused,
        encoder_ms=round(result.encoder_ms, 1),
        decoder_ms=round(result.decoder_ms, 1),
    )


def sam3_segment_response(result: ConceptResult) -> Sam3SegmentResponse:
    """Build the SAM 3 response: the union mask plus what was detected."""
    base = segment_response(result)
    return Sam3SegmentResponse(
        **base.model_dump(),
        instances=result.instances,
        instance_scores=[round(score, 4) for score in result.instance_scores],
        exemplars=[box.as_list() + [float(box.label)] for box in result.exemplars],
    )


def propagate_response(result: PropagateResult) -> PropagateResponse:
    """Build the propagation response, one entry per frame of the window."""
    return PropagateResponse(
        backend=result.backend,
        model=result.model,
        device=result.device,
        height=result.height,
        width=result.width,
        elapsed_ms=round(result.elapsed_ms, 1),
        masks=[
            PropagatedFrame(
                frame_index=item.frame_index,
                rle=rle_mask(item.mask),
                runs=_mask_runs(item.mask),
                area=item.area,
            )
            for item in result.masks
        ],
    )

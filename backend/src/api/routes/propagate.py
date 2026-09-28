"""Propagate one mask across a window of neighbouring frames."""

from __future__ import annotations

from typing import List

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.api.deps import (
    PropagateDep,
    SettingsDep,
    parse_json_field,
    read_capped_upload,
)
from src.api.serializers import propagate_response
from src.core.errors import PayloadTooLarge
from src.domain.rle import decode_rle
from src.inference.propagate import FrameInput, PropagateError
from src.schemas.propagate import PropagateResponse, PropagateStatus
from src.schemas.sam import ModelStatus

router = APIRouter(prefix="/api/propagate", tags=["propagate"])


@router.get("/status", response_model=PropagateStatus)
def propagate_status(service: PropagateDep) -> PropagateStatus:
    """Report tracker availability and the per-request frame cap."""
    status = service.status()
    return PropagateStatus(
        sam3=ModelStatus(**status["sam3"]),
        max_frames=status["max_frames"],
    )


def _parse_anchor_mask(raw: str) -> np.ndarray:
    """Decode the `mask` form field into a bool (H, W) array."""
    payload = parse_json_field(raw, "mask")
    if not isinstance(payload, dict):
        raise PropagateError("`mask` must be {size: [h, w], counts: str}.")
    try:
        size, counts = payload["size"], payload["counts"]
    except KeyError as exc:
        raise PropagateError("`mask` must be {size: [h, w], counts: str}.") from exc
    return decode_rle(size, counts)


@router.post("", response_model=PropagateResponse)
async def propagate_mask(
    settings: SettingsDep,
    service: PropagateDep,
    frames: List[UploadFile] = File(
        ..., description="The frames of the window (anchor included), any order"
    ),
    frame_indices: str = Form(
        ..., description="JSON list with the clip frame index of each uploaded file"
    ),
    anchor: int = Form(..., description="Clip frame index that carries the mask"),
    mask: str = Form(..., description='JSON RLE {"size": [h, w], "counts": str}'),
    backward: int = Form(0, description="Frames to track before the anchor"),
    forward: int = Form(0, description="Frames to track after the anchor"),
    backend: str = Form("sam3", description="Tracker to use; only 'sam3' is available"),
) -> PropagateResponse:
    """Propagate one mask over `backward` + `forward` neighbouring frames.

    The browser uploads exactly the frames of the window, so the request size is
    bounded by `PROPAGATE_MAX_FRAMES`; longer stretches take several runs. The
    anchor's own mask is not returned.
    """
    indices = parse_json_field(frame_indices, "frame_indices")
    if not isinstance(indices, list) or not all(
        isinstance(index, int) for index in indices
    ):
        raise PropagateError("`frame_indices` must be a JSON list of integers.")
    if len(indices) != len(frames):
        raise PropagateError(
            f"{len(frames)} files but {len(indices)} frame indices were sent."
        )

    anchor_mask = _parse_anchor_mask(mask)

    max_frames = settings.propagate_max_frames
    if len(frames) > max_frames:
        raise PayloadTooLarge(
            f"{len(frames)} frames exceed PROPAGATE_MAX_FRAMES={max_frames}; "
            "propagate in shorter runs."
        )

    inputs: List[FrameInput] = []
    for index, upload in zip(indices, frames):
        data = await read_capped_upload(
            upload, settings.max_frame_bytes, label=f"Frame {index}"
        )
        inputs.append(FrameInput(index=index, data=data))

    result = await run_in_threadpool(
        service.propagate,
        backend,
        inputs,
        anchor,
        anchor_mask,
        backward=backward,
        forward=forward,
    )
    return propagate_response(result)

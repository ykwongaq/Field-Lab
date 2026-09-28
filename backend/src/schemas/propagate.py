"""Mask propagation response models."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel

from src.schemas.masks import ForegroundRun, RleMask
from src.schemas.sam import ModelStatus


class PropagateStatus(BaseModel):
    """Which tracker can propagate masks, and the per-request frame cap."""

    sam3: ModelStatus
    max_frames: int


class PropagatedFrame(BaseModel):
    """The tracker's mask on one frame of the window (empty masks included)."""

    frame_index: int
    rle: RleMask
    runs: List[ForegroundRun]
    area: int


class PropagateResponse(BaseModel):
    backend: str
    model: str
    device: str
    height: int
    width: int
    elapsed_ms: float
    masks: List[PropagatedFrame]

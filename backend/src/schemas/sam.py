"""SAM 3 status and segmentation response models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from src.schemas.masks import ForegroundRun, RleMask


class ModelStatus(BaseModel):
    """Availability of a lazily loaded model."""

    available: bool
    loaded: bool
    model: str
    device: str
    error: Optional[str] = None


class Sam3Status(ModelStatus):
    """SAM 3 status plus the detection threshold in use."""

    threshold: float


class SegmentResponse(BaseModel):
    """One mask, in the three forms the reviewer needs."""

    rle: RleMask
    runs: List[ForegroundRun]
    height: int
    width: int
    score: float
    area: int
    bbox: Optional[List[int]] = None  # [x, y, w, h]
    embedding_reused: bool
    encoder_ms: float
    decoder_ms: float


class Sam3SegmentResponse(SegmentResponse):
    """Union mask of every detected instance, plus what was detected."""

    instances: int
    instance_scores: List[float]
    exemplars: List[List[float]]

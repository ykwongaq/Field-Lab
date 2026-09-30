"""SAM 3 status and segmentation DTOs.

The request is a JSON body rather than a multipart upload: frames live in the
session the clip was opened into, so a prompt only has to name the frame it
applies to.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field

from src.schemas.rle import ForegroundRun, RleMask


class ModelStatus(BaseModel):
    """Availability of a lazily loaded model."""

    available: bool
    loaded: bool
    model: str
    device: str
    error: Optional[str] = None


class Sam3Status(ModelStatus):
    """SAM 3 status, plus what is resident and what prompting is possible."""

    threshold: float
    loaded_models: List[str] = Field(default_factory=list)
    cache_entries: int = 0
    #: False when the model was loaded without instance interactivity, which is
    #: what point and box prompts need.
    point_prompts: bool = True


class PromptPointModel(BaseModel):
    """One click, in frame pixels. `label` 1 includes, 0 excludes."""

    x: float
    y: float
    label: int = 1


class PromptBoxModel(BaseModel):
    """One box, in frame pixels."""

    x0: float
    y0: float
    x1: float
    y1: float
    label: int = 1


class PromptRequest(BaseModel):
    """What the user asked for on the frame.

    `kind` decides the code path, not just the payload shape: `point` and `box`
    mean *this object* and go through the instance-interactivity predictor, while
    `text` means *this class* and goes through the concept detector.
    """

    kind: str = Field(..., description="One of: point, box, text")
    points: List[PromptPointModel] = Field(default_factory=list)
    boxes: List[Union[PromptBoxModel, List[float]]] = Field(default_factory=list)
    text: Optional[str] = None


class SegmentRequest(BaseModel):
    """Segment one frame of an open session."""

    session_id: str
    frame_index: int = Field(..., ge=0)
    prompt: PromptRequest
    #: 0 keeps every proposal; a positive number keeps the best `max_instances`.
    max_instances: int = 0


class InstanceResponse(BaseModel):
    """One proposed mask, so a caller can offer the alternatives."""

    rle: RleMask
    runs: List[ForegroundRun]
    score: float
    area: int
    bbox: Optional[List[int]] = None  # [x, y, w, h]


class SegmentResponse(BaseModel):
    """The mask to use, the alternatives behind it, and how long it took."""

    rle: RleMask
    runs: List[ForegroundRun]
    height: int
    width: int
    score: float
    area: int
    bbox: Optional[List[int]] = None
    kind: str
    prompt: str
    embedding_reused: bool
    encoder_ms: float
    decoder_ms: float
    #: Every proposal: 3 candidates for a point/box prompt, N instances for text.
    instances: List[InstanceResponse] = Field(default_factory=list)
    instance_scores: List[float] = Field(default_factory=list)

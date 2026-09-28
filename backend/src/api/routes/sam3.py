"""SAM 3 concept segmentation on a single frame."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.api.deps import Sam3Dep, SettingsDep, parse_json_field, read_capped_upload
from src.api.serializers import sam3_segment_response
from src.domain.segmentation import PromptError, parse_points
from src.schemas.sam import Sam3SegmentResponse, Sam3Status

router = APIRouter(prefix="/api/sam3", tags=["sam3"])


@router.get("/status", response_model=Sam3Status)
def sam3_status(sam3: Sam3Dep) -> Sam3Status:
    """Report whether SAM 3 is installed, loaded, and on which device."""
    return Sam3Status(**sam3.status())


@router.post("/segment", response_model=Sam3SegmentResponse)
async def sam3_segment(
    settings: SettingsDep,
    sam3: Sam3Dep,
    image: UploadFile = File(..., description="The frame image (JPEG/PNG)"),
    points: str = Form(
        "[]",
        description='JSON list of {"x","y","label"} clicks in frame pixels; '
        "may be empty when `text` is given",
    ),
    text: Optional[str] = Form(
        None, description="Optional class name / noun phrase, e.g. 'coral'"
    ),
    image_key: Optional[str] = Form(
        None,
        description="Stable id of the frame so repeated prompts reuse the cached embedding",
    ),
) -> Sam3SegmentResponse:
    """Segment a whole class on one frame from clicks and/or a class name.

    The browser calls this after every click while the reviewer refines the
    prompt; repeated calls on the same `image_key` reuse the cached embedding.
    """
    raw_points = parse_json_field(points, "points")
    if not isinstance(raw_points, list):
        raise PromptError("`points` must be a list.")
    # Unlike a click-only tool, an empty click list is fine when text is given.
    parsed_points = parse_points(raw_points) if raw_points else []
    class_name = (text or "").strip() or None
    if not parsed_points and class_name is None:
        raise PromptError("Click on an example of the class or type its name.")

    data = await read_capped_upload(image, settings.max_frame_bytes, label="Frame")
    result = await run_in_threadpool(
        sam3.segment_with_points,
        data,
        parsed_points,
        text=class_name,
        image_key=image_key,
    )
    return sam3_segment_response(result)

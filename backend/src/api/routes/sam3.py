"""SAM 3 prompts on a single frame of an open session.

The frame is named by index, not uploaded: the clip's pixels already live in the
session the archive was opened into, so a click does not pay for an image
upload, and the vision embeddings for that frame can be cached across the whole
click-by-click refinement of a mask.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import (
    SAM3_DISABLED_MESSAGE,
    ClientDep,
    Sam3Dep,
    SettingsDep,
    require_sam3,
)
from src.api.serializers import segment_response
from src.core.sessions import open_session
from src.domain.prompts import SegmentPrompt
from src.inference.frames import frame_size
from src.schemas.sam import Sam3Status, SegmentRequest, SegmentResponse

router = APIRouter(prefix="/api/sam3", tags=["sam3"])


@router.get("/status", response_model=Sam3Status)
def sam3_status(settings: SettingsDep, service: Sam3Dep) -> Sam3Status:
    """Report whether SAM 3 is installed, loaded, and on which device.

    With `sam3.enabled=false` this answers from the settings alone, so the
    service is never asked to load anything.
    """
    if not settings.enable_sam3:
        return Sam3Status(
            available=False,
            loaded=False,
            model="SAM 3",
            device=settings.sam3_device,
            error=SAM3_DISABLED_MESSAGE,
        )
    status = service.status()
    return Sam3Status(
        available=bool(status["available"]),
        loaded=bool(status["loaded"]),
        model=str(status["model"]),
        device=str(status["device"]),
        error=status.get("error"),
        loaded_models=list(status.get("loaded_models") or []),
        cache_entries=int(status.get("cache_entries") or 0),
        point_prompts=bool(status.get("point_prompts", True)),
    )


@router.post("/segment", response_model=SegmentResponse)
def sam3_segment(
    request: SegmentRequest,
    settings: SettingsDep,
    service: Sam3Dep,
    client_id: ClientDep,
) -> SegmentResponse:
    """Turn one prompt on one frame into a mask.

    `point` and `box` describe a single object; `text` describes a class and may
    match several instances, which come back in `instances` so an instance-mode
    project can split them instead of gluing them into one object.
    """
    require_sam3(settings)
    session = open_session(settings.temp_dir, request.session_id, owner=client_id)
    prompt = SegmentPrompt.from_wire(
        request.prompt.kind,
        points=[point.model_dump() for point in request.prompt.points],
        boxes=[
            box if isinstance(box, list) else box.model_dump()
            for box in request.prompt.boxes
        ],
        text=request.prompt.text,
    )
    # Check the prompt against the frame it will be applied to *before* loading a
    # model: a click outside the frame is a 422, not a wasted forward pass.
    width, height = frame_size(session, request.frame_index)
    prompt.validate(width, height)

    result = service.segment(session, request.frame_index, prompt)
    session.touch()
    return segment_response(result, max_instances=request.max_instances)

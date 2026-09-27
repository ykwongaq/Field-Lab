"""FastAPI backend for the Video Segmenter review tool.

The frontend reads a project ZIP locally (frames + annotation JSON) and uses
this service only to decode pycocotools compressed RLE segmentation masks into
foreground runs that the browser can draw directly as 1px-wide vertical strips.
"""

import json
import logging
import os
import shutil
import tempfile
from typing import List, Optional, Union

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pycocotools import mask as mask_utils
from pydantic import BaseModel

from project_builder import (
    MODES,
    VIDEO_EXTENSIONS,
    VideoOpenError,
    build_project_from_video,
    project_name_for,
)

from sam2_service import PromptError, Sam2Unavailable, parse_points
from sam2_service import service as sam2
from sam3_service import Sam3Service, Sam3Unavailable

logger = logging.getLogger("vsr.sam")
sam3 = Sam3Service()

app = FastAPI(title="Video Segmenter backend")

# The frontend talks to this service from the Vite dev origin (or a hosted
# origin), so allow cross-origin requests. No credentials are used.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    
    expose_headers=[
        "Content-Disposition",
        "X-Project-Name",
        "X-Project-Mode",
        "X-Project-Frames",
        "X-Project-Fps",
    ],
)

MAX_UPLOAD_BYTES = int(os.environ.get("VSR_MAX_UPLOAD_BYTES", str(4 * 1024**3)))
MAX_FRAME_BYTES = int(os.environ.get("VSR_MAX_FRAME_BYTES", str(32 * 1024**2)))

@app.on_event("startup")
def _maybe_warm_sam3() -> None:
    """Load SAM 3 at start-up when SAM3_EAGER=1 (otherwise on first request)."""
    if sam3.config.eager:
        try:
            sam3.warmup()
        except Sam3Unavailable as exc:  # keep the API up; status() explains
            print(f"[sam3] not available: {exc}")

@app.on_event("startup")
def _maybe_warm_sam2() -> None:
    """Optionally load SAM 2 at start-up."""
    if sam2.config.eager:
        try:
            sam2.warmup()
        except Sam2Unavailable as exc:
            print(f"[sam2] not available: {exc}")

class RleMask(BaseModel):
    """A single pycocotools RLE mask: `size` is [height, width]."""

    size: List[int]
    counts: Union[str, List[int]]


class DecodeRequest(BaseModel):
    masks: List[RleMask]


class ForegroundRun(BaseModel):
    x: int
    y: int
    length: int


class DecodedMask(BaseModel):
    height: int
    width: int
    runs: List[ForegroundRun]


class DecodeResponse(BaseModel):
    masks: List[DecodedMask]


def binary_to_runs(binary: np.ndarray) -> List[ForegroundRun]:
    """Turn a decoded binary mask into 1px-wide vertical foreground strips.

    `pycocotools.mask.decode` returns an array of shape (height, width) whose
    rows are the frame's y axis and columns are the x axis. Each column is
    scanned for contiguous foreground pixels, producing one strip per run.
    """
    height, width = binary.shape
    result: List[ForegroundRun] = []
    for x in range(width):
        column = binary[:, x]
        boundaries = np.diff(np.concatenate(([0], column, [0])))
        starts = np.flatnonzero(boundaries == 1)
        ends = np.flatnonzero(boundaries == -1)
        for start, end in zip(starts, ends):
            result.append(ForegroundRun(x=x, y=int(start), length=int(end - start)))
    return result


def decode_mask(mask: RleMask) -> DecodedMask:
    """Decode a pycocotools RLE mask into drawable foreground runs.

    Compressed counts (a base64 string) are decoded directly by
    `pycocotools.mask.decode`; uncompressed counts (a list of run lengths) are
    first normalised with `frPyObjects` into the compressed form.
    """
    height, width = mask.size
    rle = {"size": [height, width], "counts": mask.counts}
    if isinstance(mask.counts, list):
        rle = mask_utils.frPyObjects(rle, height, width)
    binary = mask_utils.decode(rle)
    return DecodedMask(height=height, width=width, runs=binary_to_runs(binary))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/decode/masks", response_model=DecodeResponse)
def decode_masks(request: DecodeRequest) -> DecodeResponse:
    decoded: List[DecodedMask] = []
    for mask in request.masks:
        if len(mask.size) != 2:
            raise HTTPException(
                status_code=422, detail="`size` must be [height, width]"
            )
        decoded.append(decode_mask(mask))
    return DecodeResponse(masks=decoded)


def _remove_tree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)

@app.post("/api/projects/create")
async def create_project(
    background: BackgroundTasks,
    video: UploadFile = File(..., description="Source video file"),
    mode: str = Form(..., description="'instance' or 'semantic'"),
    name: Optional[str] = Form(None, description="Project name"),
    frame_step: int = Form(1, ge=1, description="Keep every N-th frame"),
    jpeg_quality: int = Form(90, ge=1, le=100),
) -> FileResponse:
    """Build a project ZIP from an uploaded video and stream it back.

    The response is the archive itself (`application/zip`) with a
    `Content-Disposition` filename plus `X-Project-*` summary headers, so the
    browser can both save the file and open it immediately in the reviewer.
    """
    if mode not in MODES:
        raise HTTPException(
            status_code=422,
            detail=f"`mode` must be one of {', '.join(MODES)}",
        )

    original_name = video.filename or "upload.mp4"
    ext = os.path.splitext(original_name)[1].lower()
    if ext and ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video extension {ext!r}; expected one of "
            + ", ".join(VIDEO_EXTENSIONS),
        )

    workdir = tempfile.mkdtemp(prefix="vsr-project-")
    background.add_task(_remove_tree, workdir)

    video_path = os.path.join(workdir, "source" + (ext or ".mp4"))
    written = 0
    with open(video_path, "wb") as sink:
        while True:
            chunk = await video.read(8 * 1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if MAX_UPLOAD_BYTES and written > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Video exceeds the upload limit of {MAX_UPLOAD_BYTES} bytes",
                )
            sink.write(chunk)
    await video.close()
    if written == 0:
        raise HTTPException(status_code=422, detail="Uploaded video is empty")

    project_name = project_name_for(name, original_name)
    output_zip = os.path.join(workdir, project_name + ".zip")
    try:
        result = await run_in_threadpool(
            build_project_from_video,
            video_path,
            output_zip,
            mode=mode,
            name=project_name,
            frame_step=frame_step,
            jpeg_quality=jpeg_quality,
            source_filename=original_name,
        )
    except VideoOpenError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return FileResponse(
        output_zip,
        media_type="application/zip",
        filename=f"{result.name}.zip",
        headers={
            "X-Project-Name": result.name,
            "X-Project-Mode": result.mode,
            "X-Project-Frames": str(result.frame_count),
            "X-Project-Fps": f"{result.fps:.6g}",
        },
    )

### SAM 2

class SamStatus(BaseModel):
    available: bool
    loaded: bool
    model: str
    device: str
    error: Optional[str] = None


class SamSegmentResponse(BaseModel):
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


@app.get("/api/sam/status", response_model=SamStatus)
def sam_status() -> SamStatus:
    return SamStatus(**sam2.status())


@app.post("/api/sam/segment", response_model=SamSegmentResponse)
async def sam_segment(
    image: UploadFile = File(..., description="The frame (JPEG/PNG) to segment"),
    points: str = Form(
        ...,
        description='JSON list of clicks: [{"x": 120, "y": 80, "label": 1}, ...]; '
        "label 1 = include, 0 = exclude; frame-pixel coordinates",
    ),
    image_key: Optional[str] = Form(
        None,
        description="Stable id of the frame (e.g. '<clip>/<frame>') so repeated "
        "prompts on the same frame reuse the cached image embedding",
    ),
) -> SamSegmentResponse:
    """Run SAM 2 on one frame with point prompts and return a single mask.

    The browser calls this after every click while the reviewer refines the
    prompt.
    """
    try:
        parsed_points = parse_points(json.loads(points))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"`points` is not JSON: {exc}")
    except PromptError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    data = await image.read(MAX_FRAME_BYTES + 1)
    await image.close()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded frame is empty")
    if len(data) > MAX_FRAME_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Frame exceeds the upload limit of {MAX_FRAME_BYTES} bytes",
        )

    try:
        result = await run_in_threadpool(
            sam2.segment, data, parsed_points, image_key=image_key
        )
    except Sam2Unavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PromptError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    rle = result.rle()
    bbox = result.bbox()
    return SamSegmentResponse(
        rle=RleMask(size=rle["size"], counts=rle["counts"]),
        runs=binary_to_runs(result.mask.astype(np.uint8)),
        height=result.height,
        width=result.width,
        score=result.score,
        area=result.area,
        bbox=list(bbox) if bbox else None,
        embedding_reused=result.embedding_reused,
        encoder_ms=round(result.encoder_ms, 1),
        decoder_ms=round(result.decoder_ms, 1),
    )


class Sam3Status(SamStatus):
    """SAM 2 status fields plus the detection threshold in use."""

    threshold: float


class Sam3SegmentResponse(SamSegmentResponse):
    """Union mask of every detected instance, plus what was detected."""

    instances: int
    instance_scores: List[float]
    exemplars: List[List[float]]


@app.get("/api/sam3/status", response_model=Sam3Status)
def sam3_status() -> Sam3Status:
    return Sam3Status(**sam3.status())


@app.post("/api/sam3/segment", response_model=Sam3SegmentResponse)
async def sam3_segment(
    image: UploadFile = File(..., description="The frame image (JPEG/PNG)"),
    points: str = Form(
        "[]",
        description='JSON list of {"x", "y", "label"} clicks in frame pixels; may be empty when `text` is given',
    ),
    text: Optional[str] = Form(
        None, description="Optional class name / noun phrase, e.g. 'coral'"
    ),
    image_key: Optional[str] = Form(
        None,
        description="Stable id of the frame so repeated prompts reuse the cached embedding",
    ),
) -> Sam3SegmentResponse:
    """Run SAM 3 on one frame and return the mask of the whole class.
    """
    try:
        raw_points = json.loads(points)
        if not isinstance(raw_points, list):
            raise PromptError("`points` must be a list.")
        # Unlike SAM 2, an empty click list is fine when a class name is given.
        parsed_points = parse_points(raw_points) if raw_points else []
        text = (text or "").strip() or None
        if not parsed_points and text is None:
            raise PromptError("Click on an example of the class or type its name.")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"`points` is not JSON: {exc}")
    except PromptError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    data = await image.read(MAX_FRAME_BYTES + 1)
    await image.close()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded frame is empty")
    if len(data) > MAX_FRAME_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Frame exceeds the upload limit of {MAX_FRAME_BYTES} bytes",
        )

    refine = None
    if parsed_points and sam2.status()["available"]:

        def refine(point):  # type: ignore[no-redef]
            return sam2.segment(data, [point], image_key=image_key).bbox()

    try:
        result = await run_in_threadpool(
            sam3.segment_with_points,
            data,
            parsed_points,
            text=text,
            image_key=image_key,
            refine=refine,
        )
        rle = result.rle()
        runs = binary_to_runs(result.mask.astype(np.uint8))
    except Sam3Unavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PromptError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # CUDA OOM, version mismatch, ...
        logger.exception("SAM 3 inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"SAM 3 inference failed: {type(exc).__name__}: {exc}",
        ) from exc

    bbox = result.bbox()
    return Sam3SegmentResponse(
        rle=RleMask(size=rle["size"], counts=rle["counts"]),
        runs=runs,
        height=result.height,
        width=result.width,
        score=result.score,
        area=result.area,
        bbox=list(bbox) if bbox else None,
        embedding_reused=result.embedding_reused,
        encoder_ms=round(result.encoder_ms, 1),
        decoder_ms=round(result.decoder_ms, 1),
        instances=result.instances,
        instance_scores=[round(s, 4) for s in result.instance_scores],
        exemplars=[b.as_list() + [float(b.label)] for b in result.exemplars],
    )

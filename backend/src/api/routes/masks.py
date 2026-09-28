"""Decode pycocotools RLE masks into runs the browser can paint."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.serializers import to_foreground_runs
from src.domain.rle import binary_to_runs, decode_rle
from src.schemas.masks import DecodeRequest, DecodeResponse, DecodedMask

router = APIRouter(prefix="/api/decode", tags=["masks"])


@router.post("/masks", response_model=DecodeResponse)
def decode_masks(request: DecodeRequest) -> DecodeResponse:
    """Decode each mask into 1px-wide vertical strips.

    The viewer draws one `fillRect` per run, which is far cheaper than shipping
    a full RLE decoder to the browser.
    """
    decoded = []
    for mask in request.masks:
        binary = decode_rle(mask.size, mask.counts)
        height, width = binary.shape
        decoded.append(
            DecodedMask(
                height=height,
                width=width,
                runs=to_foreground_runs(binary_to_runs(binary)),
            )
        )
    return DecodeResponse(masks=decoded)

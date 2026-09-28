"""Mask codec DTOs, shared by every endpoint that returns a mask."""

from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel


class RleMask(BaseModel):
    """A single pycocotools RLE mask; `size` is [height, width]."""

    size: List[int]
    counts: Union[str, List[int]]


class ForegroundRun(BaseModel):
    """One 1px-wide vertical strip of a decoded mask."""

    x: int
    y: int
    length: int


class DecodeRequest(BaseModel):
    masks: List[RleMask]


class DecodedMask(BaseModel):
    height: int
    width: int
    runs: List[ForegroundRun]


class DecodeResponse(BaseModel):
    masks: List[DecodedMask]

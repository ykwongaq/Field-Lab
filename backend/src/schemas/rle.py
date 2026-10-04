"""Mask codec DTOs, shared by every endpoint that returns a mask."""

from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel


class RleMask(BaseModel):
    """A single pycocotools RLE mask; `size` is [height, width]."""

    size: List[int]
    counts: Union[str, List[int]]


class ForegroundRun(BaseModel):
    """One 1px-wide vertical strip of a decoded mask.

    Only sent where a single mask is being handed back for immediate painting
    (a segment response). Propagated frames omit it on purpose: the runs are
    computed in Python, so shipping them for hundreds of frames would cost far
    more than the client-side RLE decode it saves.
    """

    x: int
    y: int
    length: int

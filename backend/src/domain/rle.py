"""pycocotools RLE conversions.

`binary_to_runs` is the interesting one: it turns a decoded mask into the
1px-wide vertical strips the browser paints, which is how the frontend avoids
shipping a mask decoder of its own.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np
from pycocotools import mask as mask_utils

from src.core.errors import InvalidRequest

Run = Tuple[int, int, int]
"""One vertical strip as `(x, y, length)`."""


def binary_to_runs(binary: np.ndarray) -> List[Run]:
    """Turn a binary (H, W) mask into 1px-wide vertical foreground strips.

    `pycocotools.mask.decode` returns rows as y and columns as x, so each column
    is scanned for contiguous foreground pixels, producing one strip per run.
    """
    _height, width = binary.shape
    runs: List[Run] = []
    for x in range(width):
        column = binary[:, x]
        boundaries = np.diff(np.concatenate(([0], column, [0])))
        starts = np.flatnonzero(boundaries == 1)
        ends = np.flatnonzero(boundaries == -1)
        for start, end in zip(starts, ends):
            runs.append((x, int(start), int(end - start)))
    return runs


def decode_rle(size: Sequence[int], counts: Any) -> np.ndarray:
    """Decode a pycocotools RLE into a bool (H, W) mask.

    `counts` may be the compressed base64 string or the uncompressed list of run
    lengths (which pycocotools first normalises with `frPyObjects`).
    """
    try:
        height, width = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise InvalidRequest("`size` must be [height, width]") from exc

    if isinstance(counts, str):
        encoded: Mapping[str, Any] = {
            "size": [height, width],
            "counts": counts.encode("ascii"),
        }
    elif isinstance(counts, (list, tuple)):
        encoded = mask_utils.frPyObjects(
            {"size": [height, width], "counts": list(counts)}, height, width
        )
    else:
        raise InvalidRequest("`counts` must be a string or a list of run lengths.")

    return mask_utils.decode(encoded).astype(bool)


def encode_rle(mask: np.ndarray) -> Tuple[List[int], str]:
    """Encode a bool (H, W) mask as `([height, width], compressed counts)`."""
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    height, width = mask.shape
    return [int(height), int(width)], counts

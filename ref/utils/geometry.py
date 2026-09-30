"""Box/mask conversions and mask-versus-box metrics.

Conventions
-----------
``xyxy``   pixel space ``(x1, y1, x2, y2)``. Every public function takes this
           unless its name says otherwise.
``xywh``   ``[x, y, w, h]`` normalised to 0~1. This is what the SAM3 *video*
           prompt takes (``add_prompt(bounding_boxes=...)``).
``cxcywh`` ``[cx, cy, w, h]`` normalised to 0~1. This is what the SAM3 *image*
           processor takes (``Sam3Processor.add_geometric_prompt``).

Masks are 2-D boolean arrays of shape ``(height, width)``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

BoxXYXY = Tuple[float, float, float, float]


def clip_box(box: Sequence[float], width: int, height: int) -> BoxXYXY:
    """Clamp ``box`` into ``[0, width] x [0, height]``, ordering the corners."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return (x1, y1, x2, y2)


def box_xyxy_to_xywh_norm(box: Sequence[float], width: int, height: int) -> list[float]:
    """``xyxy`` pixels -> ``[x, y, w, h]`` normalised 0~1 (SAM3 video prompt)."""
    x1, y1, x2, y2 = clip_box(box, width, height)
    return [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]


def box_xyxy_to_cxcywh_norm(
    box: Sequence[float], width: int, height: int
) -> list[float]:
    """``xyxy`` pixels -> ``[cx, cy, w, h]`` normalised 0~1 (SAM3 image prompt)."""
    x1, y1, x2, y2 = clip_box(box, width, height)
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    return [cx, cy, (x2 - x1) / width, (y2 - y1) / height]


def box_xywh_norm_to_xyxy(box: Sequence[float], width: int, height: int) -> BoxXYXY:
    """``[x, y, w, h]`` normalised 0~1 -> ``xyxy`` pixels."""
    x, y, w, h = (float(v) for v in box)
    return (
        x * width,
        y * height,
        (x + w) * width,
        (y + h) * height,
    )


def box_area(box: Sequence[float]) -> float:
    """Area of an ``xyxy`` box (0 for a degenerate one)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU between two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def as_bool_mask(mask) -> np.ndarray:
    """Return ``mask`` as a 2-D boolean array (no copy when already bool)."""
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def _int_bounds(box: Sequence[float], mask: np.ndarray) -> Optional[Tuple[int, ...]]:
    """Clip an ``xyxy`` box to the mask grid, ``None`` when degenerate."""
    x1, y1, x2, y2 = (int(v) for v in box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(mask.shape[1], x2)
    y2 = min(mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def mask_area(mask) -> int:
    """Number of set pixels."""
    return int(as_bool_mask(mask).sum())


def is_usable_mask(mask, min_area: int = 1) -> bool:
    """True when ``mask`` is a 2-D array with at least ``min_area`` pixels set."""
    if mask is None:
        return False
    try:
        arr = as_bool_mask(mask)
    except ValueError:
        return False
    return int(arr.sum()) >= max(1, int(min_area))


def mask_box_intersection(mask, box: Sequence[float]) -> int:
    """Number of mask pixels inside ``box``."""
    arr = as_bool_mask(mask)
    bounds = _int_bounds(box, arr)
    if bounds is None:
        return 0
    x1, y1, x2, y2 = bounds
    return int(arr[y1:y2, x1:x2].sum())


def mask_box_containment(mask, box: Sequence[float]) -> float:
    """``|mask AND box| / |mask|`` -- how much of the mask sits inside the box.

    This is the anchor sanity check: a probe mask for *this* track should sit
    inside *this* track's ground-truth box. A low value means the probe latched
    onto something else (a different instance, or background).
    """
    total = mask_area(mask)
    if total == 0:
        return 0.0
    return mask_box_intersection(mask, box) / total


def mask_box_iou(mask, box: Sequence[float]) -> float:
    """IoU between a binary mask and an ``xyxy`` box treated as a filled region.

    Equivalent to building a full ``box_mask`` and taking the logical IoU, but
    only slices the box window (cheaper for 1080x1920 frames).
    """
    arr = as_bool_mask(mask)
    total = int(arr.sum())
    if total == 0:
        return 0.0
    bounds = _int_bounds(box, arr)
    if bounds is None:
        return 0.0
    x1, y1, x2, y2 = bounds
    inter = int(arr[y1:y2, x1:x2].sum())
    if inter == 0:
        return 0.0
    box_pixels = (x2 - x1) * (y2 - y1)
    union = total + box_pixels - inter
    return inter / union if union > 0 else 0.0


def mask_centroid(mask) -> Optional[Tuple[float, float]]:
    """``(x, y)`` centroid of the mask, or ``None`` when it is empty."""
    arr = as_bool_mask(mask)
    ys, xs = np.nonzero(arr)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def ellipse_mask(
    box: Sequence[float], width: int, height: int, inset: float = 0.0
) -> np.ndarray:
    """Boolean mask of the ellipse inscribed in ``box``.

    ``inset`` shrinks both radii by that fraction (0.0 = box-filling ellipse).
    Used as a *synthetic* anchor when no probe mask is available: a filled box
    would claim background pixels, whereas an inscribed ellipse is a
    conservative core prior.

    The comparison is strict (``< 1``), so the box's exclusive right/bottom edge
    is never claimed and ``mask_box_containment(mask, box)`` stays exactly 1.0.
    """
    arr = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = clip_box(box, width, height)
    if x2 <= x1 or y2 <= y1:
        return arr
    inset = min(max(float(inset), 0.0), 0.99)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx = max((x2 - x1) / 2.0 * (1.0 - inset), 0.5)
    ry = max((y2 - y1) / 2.0 * (1.0 - inset), 0.5)
    yy, xx = np.ogrid[:height, :width]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0


def filled_box_mask(box: Sequence[float], width: int, height: int) -> np.ndarray:
    """Boolean mask of the filled ``box`` (mostly for tests/comparisons)."""
    arr = np.zeros((height, width), dtype=bool)
    bounds = _int_bounds(box, arr)
    if bounds is None:
        return arr
    x1, y1, x2, y2 = bounds
    arr[y1:y2, x1:x2] = True
    return arr

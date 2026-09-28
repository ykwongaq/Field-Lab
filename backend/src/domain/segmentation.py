"""Value objects for click- and text-prompted segmentation.

These types are the shared vocabulary between the API layer and the inference
services, so neither has to import the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidRequest

POSITIVE = 1
NEGATIVE = 0


class PromptError(InvalidRequest):
    """The click or text prompt is malformed."""


@dataclass(frozen=True)
class PromptPoint:
    """One click in frame-pixel coordinates. `label` 1 = include, 0 = exclude."""

    x: float
    y: float
    label: int

    def validate(self, width: int, height: int) -> None:
        if self.label not in (POSITIVE, NEGATIVE):
            raise PromptError(f"point label must be 0 or 1, got {self.label!r}")
        if not (0 <= self.x < width and 0 <= self.y < height):
            raise PromptError(
                f"point ({self.x:.1f}, {self.y:.1f}) is outside the {width}x{height} frame"
            )


@dataclass(frozen=True)
class ExemplarBox:
    """One exemplar box in frame pixels, `label` 1 = example, 0 = counter-example."""

    x0: float
    y0: float
    x1: float
    y1: float
    label: int

    def validate(self, width: int, height: int) -> None:
        if self.label not in (POSITIVE, NEGATIVE):
            raise PromptError(f"box label must be 0 or 1, got {self.label!r}")
        if not (0 <= self.x0 < self.x1 <= width and 0 <= self.y0 < self.y1 <= height):
            raise PromptError(
                f"box ({self.x0:.0f}, {self.y0:.0f})-({self.x1:.0f}, {self.y1:.0f}) "
                f"is not inside the {width}x{height} frame"
            )

    def as_list(self) -> List[float]:
        return [float(self.x0), float(self.y0), float(self.x1), float(self.y1)]


@dataclass
class SegmentResult:
    """One mask produced by a segmentation model."""

    mask: np.ndarray
    score: float
    height: int
    width: int
    encoder_ms: float
    decoder_ms: float
    embedding_reused: bool

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """`(x, y, w, h)` of the mask, or `None` when the mask is empty."""
        ys, xs = np.nonzero(self.mask)
        if xs.size == 0:
            return None
        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )


@dataclass
class ConceptResult(SegmentResult):
    """Union of every detected instance of a concept, plus per-instance detail."""

    instances: int = 0
    instance_scores: List[float] = field(default_factory=list)
    exemplars: List[ExemplarBox] = field(default_factory=list)


def parse_points(raw: Any) -> List[PromptPoint]:
    """Validate the JSON-decoded `points` form field."""
    if not isinstance(raw, list) or not raw:
        raise PromptError("`points` must be a non-empty list.")
    points: List[PromptPoint] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PromptError(f"points[{index}] must be an object.")
        try:
            x = float(item["x"])
            y = float(item["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PromptError(f"points[{index}] needs numeric x and y.") from exc
        label = item.get("label", POSITIVE)
        if isinstance(label, str):
            label = {"positive": POSITIVE, "negative": NEGATIVE}.get(label.lower(), -1)
        elif isinstance(label, bool):
            label = POSITIVE if label else NEGATIVE
        points.append(PromptPoint(x=x, y=y, label=int(label)))
    return points


def exemplar_boxes_from_points(
    points: Sequence[PromptPoint],
    width: int,
    height: int,
    *,
    fraction: float = 0.06,
) -> List[ExemplarBox]:
    """Turn clicks into square exemplar boxes sized as `fraction` of the frame.

    Used only when a click is given without a text prompt: SAM 3 wants boxes,
    the user gives points, so each click becomes a small box around itself.
    """
    boxes: List[ExemplarBox] = []
    side = max(8.0, fraction * min(width, height))
    for point in points:
        point.validate(width, height)
        half = side / 2
        x0 = max(0.0, point.x - half)
        y0 = max(0.0, point.y - half)
        x1 = min(float(width), point.x + half)
        y1 = min(float(height), point.y + half)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        boxes.append(ExemplarBox(x0, y0, x1, y1, point.label))
    return boxes

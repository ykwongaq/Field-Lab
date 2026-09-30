"""What a user can say about a frame, and what comes back.

Every image prompt compiles down to one currency — a binary mask — so the rest of
the system (drafts, review, propagation) never needs to know which kind of prompt
produced it. That is the whole point of this module: the prompt types are a thin
input vocabulary, and everything downstream sees masks.

Three kinds, deliberately kept apart:

``text``
    A concept ("shark"), which may match several instances. The union of those
    instances is a legitimate class mask in semantic projects and must be split
    into separate objects in instance projects.
``box``
    One object inside a rectangle, optionally qualified by text.
``point``
    One object under the click(s). This needs SAM 3's instance-interactivity
    predictor, not the concept detector, because "segment this object" and
    "segment everything like this" are different questions.

Coordinates are always frame pixels when they arrive here; translating to the
normalised forms SAM 3 wants happens in the inference layer, so the API and the
UI can keep speaking pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidRequest

#: Prompt kinds.
KIND_POINT = "point"
KIND_BOX = "box"
KIND_TEXT = "text"
KINDS = (KIND_POINT, KIND_BOX, KIND_TEXT)

#: Click labels, matching SAM's convention.
POSITIVE = 1
NEGATIVE = 0

#: A box smaller than this in either direction is a mis-drag, not a prompt.
MIN_BOX_PIXELS = 2.0

#: Longest accepted text prompt.
MAX_TEXT_LENGTH = 120


class PromptError(InvalidRequest):
    """The prompt is malformed or describes nothing we can act on."""


def _coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromptError(f"`{field_name}` must be a number, got {value!r}.")
    return float(value)


def _coerce_label(value: Any) -> int:
    if value is None:
        return POSITIVE
    if isinstance(value, bool):
        return POSITIVE if value else NEGATIVE
    if isinstance(value, str):
        mapped = {"positive": POSITIVE, "negative": NEGATIVE}.get(value.strip().lower())
        if mapped is None:
            raise PromptError(f"A point label must be 0/1, got {value!r}.")
        return mapped
    if isinstance(value, int) and value in (POSITIVE, NEGATIVE):
        return value
    raise PromptError(f"A point label must be 0 or 1, got {value!r}.")


@dataclass(frozen=True)
class PromptPoint:
    """One click in frame pixels. `label` 1 = include, 0 = exclude."""

    x: float
    y: float
    label: int = POSITIVE

    @classmethod
    def from_wire(cls, item: Any, *, index: int) -> "PromptPoint":
        if not isinstance(item, dict):
            raise PromptError(f"points[{index}] must be an object with x and y.")
        try:
            x = _coerce_float(item["x"], field_name=f"points[{index}].x")
            y = _coerce_float(item["y"], field_name=f"points[{index}].y")
        except KeyError as exc:
            raise PromptError(f"points[{index}] needs both x and y.") from exc
        return cls(x=x, y=y, label=_coerce_label(item.get("label")))

    def validate(self, width: int, height: int) -> None:
        if not (0 <= self.x < width and 0 <= self.y < height):
            raise PromptError(
                f"Point ({self.x:.1f}, {self.y:.1f}) is outside the "
                f"{width}x{height} frame."
            )

    def as_xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class PromptBox:
    """One box in frame pixels. `label` 1 = include, 0 = exclude."""

    x0: float
    y0: float
    x1: float
    y1: float
    label: int = POSITIVE

    @classmethod
    def from_wire(cls, item: Any, *, index: int) -> "PromptBox":
        """Accept `[x0, y0, x1, y1]` or `{x0, y0, x1, y1, label}`."""
        if isinstance(item, dict):
            try:
                values = [
                    _coerce_float(item[key], field_name=f"boxes[{index}].{key}")
                    for key in ("x0", "y0", "x1", "y1")
                ]
            except KeyError as exc:
                raise PromptError(
                    f"boxes[{index}] needs x0, y0, x1 and y1."
                ) from exc
            label = _coerce_label(item.get("label"))
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            values = [
                _coerce_float(value, field_name=f"boxes[{index}][{position}]")
                for position, value in enumerate(item)
            ]
            label = POSITIVE
        else:
            raise PromptError(
                f"boxes[{index}] must be [x0, y0, x1, y1] or an object with those keys."
            )
        return cls(*values, label=label)

    def __post_init__(self) -> None:
        # Order the corners so a backwards drag is still a box.
        if self.x1 < self.x0 or self.y1 < self.y0:
            object.__setattr__(self, "x0", min(self.x0, self.x1))
            object.__setattr__(self, "x1", max(self.x0, self.x1))
            object.__setattr__(self, "y0", min(self.y0, self.y1))
            object.__setattr__(self, "y1", max(self.y0, self.y1))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def validate(self, width: int, height: int) -> None:
        if self.width < MIN_BOX_PIXELS or self.height < MIN_BOX_PIXELS:
            raise PromptError(
                f"The box is too small ({self.width:.0f}x{self.height:.0f} px); "
                f"drag at least {MIN_BOX_PIXELS:.0f} px in each direction."
            )
        if not (0 <= self.x0 and 0 <= self.y0 and self.x1 <= width and self.y1 <= height):
            raise PromptError(
                f"The box ({self.x0:.0f}, {self.y0:.0f})-({self.x1:.0f}, {self.y1:.0f}) "
                f"is not inside the {width}x{height} frame."
            )

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_cxcywh_normalized(self, width: int, height: int) -> List[float]:
        """`[cx, cy, w, h]` in 0~1 — what the SAM 3 image processor wants."""
        return [
            (self.x0 + self.x1) / 2.0 / width,
            (self.y0 + self.y1) / 2.0 / height,
            self.width / width,
            self.height / height,
        ]


@dataclass(frozen=True)
class SegmentPrompt:
    """One validated request to turn a prompt into a mask."""

    kind: str
    points: Tuple[PromptPoint, ...] = ()
    boxes: Tuple[PromptBox, ...] = ()
    text: Optional[str] = None

    @classmethod
    def from_wire(
        cls,
        kind: str,
        *,
        points: Optional[Iterable[Any]] = None,
        boxes: Optional[Iterable[Any]] = None,
        text: Optional[str] = None,
    ) -> "SegmentPrompt":
        """Build a prompt from JSON-shaped input, validating as it goes."""
        if kind not in KINDS:
            raise PromptError(
                f"`kind` must be one of {', '.join(KINDS)}, got {kind!r}."
            )

        parsed_points = tuple(
            PromptPoint.from_wire(item, index=index)
            for index, item in enumerate(points or ())
        )
        parsed_boxes = tuple(
            PromptBox.from_wire(item, index=index) for index, item in enumerate(boxes or ())
        )
        label = (text or "").strip() or None
        if label is not None and len(label) > MAX_TEXT_LENGTH:
            raise PromptError(
                f"The text prompt is longer than {MAX_TEXT_LENGTH} characters."
            )

        if kind == KIND_POINT:
            if not parsed_points:
                raise PromptError("A point prompt needs at least one click.")
            if label is not None:
                raise PromptError(
                    "A point prompt segments one object; combine it with a text prompt "
                    "only through the box tool."
                )
        elif kind == KIND_BOX:
            if not parsed_boxes:
                raise PromptError("A box prompt needs at least one box.")
            if parsed_points:
                raise PromptError(
                    "Send clicks as a point prompt; a box prompt carries boxes only."
                )
        else:  # KIND_TEXT
            if label is None:
                raise PromptError("A text prompt needs some text.")
            if parsed_points or parsed_boxes:
                raise PromptError(
                    "A text prompt describes a concept; drop the clicks and boxes."
                )

        return cls(kind=kind, points=parsed_points, boxes=parsed_boxes, text=label)

    def validate(self, width: int, height: int) -> None:
        """Check every coordinate against the frame it will be applied to."""
        for point in self.points:
            point.validate(width, height)
        for box in self.boxes:
            box.validate(width, height)
        if self.kind == KIND_TEXT and not self.text:
            raise PromptError("A text prompt needs some text.")

    @property
    def positive_points(self) -> Tuple[PromptPoint, ...]:
        return tuple(point for point in self.points if point.label == POSITIVE)

    @property
    def primary_box(self) -> Optional[PromptBox]:
        """The box to use when only one can be sent (the image processor takes one)."""
        included = [box for box in self.boxes if box.label == POSITIVE]
        if not included:
            return self.boxes[0] if self.boxes else None
        return max(included, key=lambda box: box.area)

    def describe(self) -> str:
        """Short human-readable summary, for logs and job records."""
        if self.kind == KIND_TEXT:
            return f"text {self.text!r}"
        if self.kind == KIND_BOX:
            return f"{len(self.boxes)} box(es)" + (
                f" + text {self.text!r}" if self.text else ""
            )
        positives = len(self.positive_points)
        negatives = len(self.points) - positives
        return (
            f"{positives} point(s)"
            + (f", {negatives} excluded" if negatives else "")
            + (f" + {len(self.boxes)} box(es)" if self.boxes else "")
        )


@dataclass
class InstanceMask:
    """One mask SAM 3 proposed, with the score it attached to it."""

    mask: np.ndarray
    score: float

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """`(x, y, w, h)` of the mask, or `None` when it is empty."""
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
class SegmentResult:
    """Masks for one prompt: the one to use, plus the alternatives behind it.

    `mask` is what the reviewer drops into a draft: the best candidate for a
    point prompt, the union for a text prompt. `instances` keeps the individual
    proposals so an instance-mode project can split a concept into objects
    instead of gluing them into one tracklet.
    """

    mask: np.ndarray
    instances: List[InstanceMask] = field(default_factory=list)
    height: int = 0
    width: int = 0
    kind: str = KIND_TEXT
    prompt: str = ""
    encoder_ms: float = 0.0
    decoder_ms: float = 0.0
    embedding_reused: bool = False

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def score(self) -> float:
        """The best instance score (0.0 when nothing was found)."""
        return max((item.score for item in self.instances), default=0.0)

    @property
    def instance_scores(self) -> List[float]:
        return [item.score for item in self.instances]

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def bbox(self) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.nonzero(self.mask)
        if xs.size == 0:
            return None
        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )


def union_of(masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Logical OR of several masks, as a fresh `shape` array."""
    union = np.zeros(shape, dtype=bool)
    for mask in masks:
        union |= mask
    return union

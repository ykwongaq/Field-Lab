"""Accept/reject policy for keyframe mask anchors.

A keyframe anchor is a mask injected into the video tracker with
``add_mask``/``add_tracker_new_mask`` for an *existing* ``obj_id``. Because the
anchor is written into tracker memory as authoritative, a bad anchor is worse
than no anchor -- it can snap the track onto a different animal.

The policy below is therefore deliberately conservative: a probe result only
becomes an anchor when it is confident *and* spatially consistent with the same
track's ground-truth box on that frame. Everything else is recorded as a
rejection with a reason so the decisions can be audited after a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .geometry import BoxXYXY, as_bool_mask, mask_area, mask_box_containment

# Rejection reasons, in the order they are checked.
REASON_OK = "ok"
REASON_NO_MASK = "no_mask"
REASON_EMPTY_MASK = "empty_mask"
REASON_HUGE_MASK = "huge_mask"
REASON_NO_SCORE = "no_score"
REASON_LOW_SCORE = "low_score"
REASON_BOX_MISMATCH = "box_mismatch"
REASON_OVER_CAP = "over_cap"


@dataclass(frozen=True)
class AnchorConfig:
    """Thresholds for accepting a keyframe probe as an anchor.

    ``min_score``        probe confidence floor (``None`` scores are rejected).
    ``min_containment``  minimum ``|mask AND gt_box| / |mask|``.
    ``max_anchors``      keep at most this many anchors per track per window.
    ``min_area``         reject near-empty masks.
    ``max_area_frac``    reject masks covering more than this fraction of the
                         frame (a probe that returned "the whole image").
    """

    min_score: float = 0.5
    min_containment: float = 0.9
    max_anchors: int = 3
    min_area: int = 16
    max_area_frac: float = 0.9

    def __post_init__(self) -> None:
        if self.max_anchors < 0:
            raise ValueError("max_anchors must be >= 0")
        if self.min_area < 0:
            raise ValueError("min_area must be >= 0")
        if not 0.0 <= self.max_area_frac <= 1.0:
            raise ValueError("max_area_frac must be in [0, 1]")
        if not 0.0 <= self.min_containment <= 1.0:
            raise ValueError("min_containment must be in [0, 1]")


@dataclass
class AnchorCandidate:
    """One probe result awaiting the accept/reject decision."""

    frame_index: int
    mask: object  # HxW array-like or None
    score: Optional[float]
    gt_box: BoxXYXY  # this track's box on the same frame


@dataclass
class Anchor:
    """An accepted keyframe anchor."""

    frame_index: int
    mask: np.ndarray
    score: float
    containment: float
    area: int


@dataclass
class AnchorDecision:
    """The outcome for one candidate, kept for logging/auditing."""

    frame_index: int
    accepted: bool
    reason: str
    score: Optional[float] = None
    containment: Optional[float] = None
    area: Optional[int] = None

    def to_json(self) -> Dict[str, object]:
        return {
            "frame_index": int(self.frame_index),
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "score": None if self.score is None else float(self.score),
            "containment": (
                None if self.containment is None else float(self.containment)
            ),
            "area": None if self.area is None else int(self.area),
        }


def evaluate_anchor(
    candidate: AnchorCandidate,
    config: AnchorConfig,
    width: int,
    height: int,
) -> Tuple[Optional[Anchor], AnchorDecision]:
    """Apply the anchor policy to one candidate.

    Returns ``(anchor_or_None, decision)``. The decision always carries the
    measured numbers so a rejected anchor can be diagnosed later.
    """
    frame = int(candidate.frame_index)
    score = None if candidate.score is None else float(candidate.score)

    if candidate.mask is None:
        return None, AnchorDecision(frame, False, REASON_NO_MASK, score=score)

    try:
        mask = as_bool_mask(candidate.mask)
    except ValueError:
        return None, AnchorDecision(frame, False, REASON_NO_MASK, score=score)

    area = mask_area(mask)
    decision = AnchorDecision(frame, False, REASON_EMPTY_MASK, score=score, area=area)

    if area < max(1, int(config.min_area)):
        return None, decision

    frame_area = max(1, int(width) * int(height))
    if area > config.max_area_frac * frame_area:
        decision.reason = REASON_HUGE_MASK
        return None, decision

    if score is None:
        decision.reason = REASON_NO_SCORE
        return None, decision
    if score < config.min_score:
        decision.reason = REASON_LOW_SCORE
        return None, decision

    containment = mask_box_containment(mask, candidate.gt_box)
    decision.containment = containment
    if containment < config.min_containment:
        decision.reason = REASON_BOX_MISMATCH
        return None, decision

    decision.accepted = True
    decision.reason = REASON_OK
    anchor = Anchor(
        frame_index=frame,
        mask=mask,
        score=score,
        containment=containment,
        area=area,
    )
    return anchor, decision


def select_anchors(
    candidates: Iterable[AnchorCandidate],
    config: AnchorConfig,
    width: int,
    height: int,
) -> Tuple[List[Anchor], List[AnchorDecision]]:
    """Evaluate every candidate and cap the accepted set to ``max_anchors``.

    When more candidates pass than ``max_anchors``, the highest-scoring ones
    win (ties broken by the lower frame index). The dropped candidates get an
    ``over_cap`` decision so the log still explains them.

    Returns ``(anchors_sorted_by_frame, decisions_in_input_order)``.
    """
    candidates = list(candidates)
    decisions: List[AnchorDecision] = []
    survivors: List[Tuple[Anchor, int]] = []

    for index, candidate in enumerate(candidates):
        anchor, decision = evaluate_anchor(candidate, config, width, height)
        decisions.append(decision)
        if anchor is not None:
            survivors.append((anchor, index))

    if len(survivors) > config.max_anchors:
        ranked = sorted(
            survivors, key=lambda pair: (-pair[0].score, pair[0].frame_index)
        )
        keep = ranked[: config.max_anchors]
        for _anchor, index in ranked[config.max_anchors :]:
            decisions[index].accepted = False
            decisions[index].reason = REASON_OVER_CAP
        survivors = keep

    anchors = sorted((anchor for anchor, _ in survivors), key=lambda a: a.frame_index)
    return anchors, decisions

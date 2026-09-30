"""Shared helpers for the MammAlps-S2 SAM3 keyframe-anchoring pipeline.

The package is deliberately dependency-light: only ``numpy`` is imported at
module level. Everything that needs ``torch``/``sam3`` (i.e. the keyframe probe
backends) imports lazily inside its own class, so the pure logic here can be
unit-tested without a GPU or the SAM3 checkout.

Modules
-------
``geometry``   box/mask conversions and mask-vs-box metrics.
``keyframes``  picking the first/middle/last *detected* frame of a window.
``anchors``    the accept/reject policy for keyframe mask anchors.
``cache``      on-disk storage for probe results and anchor decisions.
``probe``      keyframe mask probes (image model, or a 1-frame video session).
"""

from .anchors import (
    Anchor,
    AnchorCandidate,
    AnchorConfig,
    AnchorDecision,
    evaluate_anchor,
    select_anchors,
)
from .cache import (
    anchor_cache_path,
    append_jsonl,
    json_rle_to_mask,
    load_anchor_cache,
    mask_to_json_rle,
    save_anchor_cache,
)
from .geometry import (
    box_xywh_norm_to_xyxy,
    box_xyxy_to_cxcywh_norm,
    box_xyxy_to_xywh_norm,
    clip_box,
    ellipse_mask,
    filled_box_mask,
    is_usable_mask,
    mask_area,
    mask_box_containment,
    mask_box_intersection,
    mask_box_iou,
    mask_centroid,
)
from .keyframes import position_count, select_keyframes, unique_sorted
from .planner import KeyframeAnchorPlanner
from .probe import (
    ImageModelProbe,
    ProbeResult,
    VideoModelProbe,
    build_probe,
    group_queries_by_text,
    pick_best_object,
)
from .tracking import outputs_to_mask_track, prompt_object_id, run_track

__all__ = [
    "Anchor",
    "AnchorCandidate",
    "AnchorConfig",
    "AnchorDecision",
    "ImageModelProbe",
    "KeyframeAnchorPlanner",
    "ProbeResult",
    "VideoModelProbe",
    "anchor_cache_path",
    "append_jsonl",
    "box_xywh_norm_to_xyxy",
    "box_xyxy_to_cxcywh_norm",
    "box_xyxy_to_xywh_norm",
    "build_probe",
    "clip_box",
    "ellipse_mask",
    "evaluate_anchor",
    "filled_box_mask",
    "group_queries_by_text",
    "is_usable_mask",
    "json_rle_to_mask",
    "load_anchor_cache",
    "mask_area",
    "mask_box_containment",
    "mask_box_intersection",
    "mask_box_iou",
    "mask_centroid",
    "mask_to_json_rle",
    "outputs_to_mask_track",
    "pick_best_object",
    "position_count",
    "prompt_object_id",
    "run_track",
    "save_anchor_cache",
    "select_anchors",
    "select_keyframes",
    "unique_sorted",
]

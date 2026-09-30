"""SAM3 session-level helpers for tracking one prompt through a video window.

These encapsulate the request sequence that
``Sam3VideoPredictorMultiGPU.handle_request`` / ``handle_stream_request``
expect, including the optional keyframe anchors:

1. ``reset_session``      - a text/box prompt restarts SAM3's semantic state.
2. ``add_prompt``         - text + box creates the object on the prompt frame.
3. ``add_mask`` (xN)      - condition that object's tracker memory on a keyframe.
4. ``propagate_in_video`` - with ``force_tracker_propagation`` when anchored.

Keeping this here (rather than inline in ``1_sam3_inference.py``) makes the
request sequence testable with a stub predictor and no GPU.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import mask_box_iou
from .probe import pick_best_mask


def prompt_object_id(
    prompt_response: Optional[Dict[str, Any]],
    prompt_box_xyxy: Optional[Sequence[float]],
) -> Optional[int]:
    """Recover the tracker ``obj_id`` that a box prompt just created.

    ``add_mask`` needs the object id, but it only exists once ``add_prompt`` has
    run, so the id is taken from the masks the prompt produced: the object whose
    mask best overlaps the prompt box is the one the prompt was aimed at.

    Returns ``None`` when the prompt produced nothing usable.
    """
    outputs = (prompt_response or {}).get("outputs") or {}
    masks = outputs.get("out_binary_masks")
    obj_ids = outputs.get("out_obj_ids")
    if masks is None or obj_ids is None or len(masks) == 0 or prompt_box_xyxy is None:
        return None
    best = pick_best_mask(masks, prompt_box_xyxy)
    if best is None or best >= len(obj_ids):
        return None
    return int(obj_ids[best])


def run_track(
    predictor,
    session_id: str,
    prompt_frame: int,
    species: str,
    box_xywh_norm: Sequence[float],
    prompt_box_xyxy: Optional[Sequence[float]] = None,
    anchor_masks: Optional[Sequence[Tuple[int, Any]]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Prompt one track (text + box) and propagate it through the whole video.

    ``anchor_masks`` is an optional list of ``(local_frame_index, mask)``
    keyframe anchors. They are injected with SAM3's ``add_mask`` request for the
    object the box prompt just created, which conditions the tracker memory on
    that frame. An anchor is authoritative and does not reset the session, so
    all of them must describe the same object.

    Returns ``{frame_index: outputs}`` with the same ``outputs`` dict shape as
    SAM3's ``propagate_in_video`` stream entries.
    """
    # Each text/box prompt restarts SAM3's semantic state, so reset first.
    predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
    prompt_response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=prompt_frame,
            text=species,
            bounding_boxes=[box_xywh_norm],  # [x, y, w, h], normalized 0~1
            bounding_box_labels=[1],  # 1 = positive box
        )
    )

    used_anchors = 0
    if anchor_masks:
        obj_id = prompt_object_id(prompt_response, prompt_box_xyxy)
        if obj_id is not None:
            for local_frame, mask in anchor_masks:
                predictor.handle_request(
                    request=dict(
                        type="add_mask",
                        session_id=session_id,
                        frame_index=int(local_frame),
                        obj_id=int(obj_id),
                        mask=mask,
                    )
                )
                used_anchors += 1

    outputs_per_frame: Dict[int, Dict[str, Any]] = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="both",
            # Anchoring registers a refinement in the action history, which
            # would propagate only that object; a forced pass covers the video.
            force_tracker_propagation=bool(used_anchors),
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def outputs_to_mask_track(
    outputs_per_frame: Dict[int, Dict[str, Any]],
    num_frames: int,
    prompt_frame: int,
    prompt_box_xyxy: Sequence[float],
) -> Dict[int, np.ndarray]:
    """Select the best object of one SAM3 session and return its masks.

    Returns ``{frame_idx: (H, W) bool ndarray}`` for the predicted object whose
    mask at ``prompt_frame`` best overlaps ``prompt_box_xyxy`` (pixels), or an
    empty dict when no object matches the prompt.
    """
    # obj_id -> {frame_index: (H, W) bool mask}
    object_frames: Dict[int, Dict[int, np.ndarray]] = {}
    for frame_idx in range(num_frames):
        outputs = outputs_per_frame.get(frame_idx)
        if outputs is None:
            continue
        masks = outputs.get("out_binary_masks")
        obj_ids = outputs.get("out_obj_ids")
        if masks is None or obj_ids is None or len(masks) == 0:
            continue
        for obj_id, mask in zip(obj_ids, masks):
            mask = np.asarray(mask, dtype=bool)
            if mask.any():
                object_frames.setdefault(int(obj_id), {})[frame_idx] = mask

    best_obj_id = None
    best_iou = -1.0
    for obj_id, frame_masks in object_frames.items():
        mask = frame_masks.get(prompt_frame)
        if mask is None:
            continue
        iou = mask_box_iou(mask, prompt_box_xyxy)
        if iou > best_iou:
            best_iou = iou
            best_obj_id = obj_id

    if best_obj_id is None:
        return {}
    return object_frames[best_obj_id]

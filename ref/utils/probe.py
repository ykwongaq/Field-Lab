"""Keyframe mask probes.

A probe answers one question: *on this keyframe, what mask does SAM3 produce for
this instance?* The result feeds the anchor policy in :mod:`utils.anchors`.

Two interchangeable backends, both returning :class:`ProbeResult`:

``ImageModelProbe``
    ``build_sam3_image_model`` + ``Sam3Processor`` (text prompt + box geometric
    prompt). One backbone pass per image, then one grounding pass per query.

``VideoModelProbe``
    A one-frame session on an *existing* SAM3 video predictor. Reuses the model
    that is already resident, so it costs no extra VRAM, but needs a session per
    keyframe.

``torch``/``sam3`` are imported lazily inside the backends, so this module (and
its pure helpers) can be imported and unit-tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cache import json_rle_to_mask, mask_to_json_rle
from .geometry import (
    BoxXYXY,
    as_bool_mask,
    box_iou,
    box_xyxy_to_cxcywh_norm,
    box_xyxy_to_xywh_norm,
    mask_box_iou,
)

#: A probe query: ``(text prompt, pixel-space xyxy box)``.
Query = Tuple[str, BoxXYXY]


@dataclass
class ProbeResult:
    """The outcome of probing one (track, keyframe) pair."""

    frame_index: int
    ok: bool
    reason: str
    mask: Optional[np.ndarray] = None
    score: float = 0.0
    box_xyxy: Optional[BoxXYXY] = None
    logits: Optional[np.ndarray] = None

    def to_cache_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form (mask as COCO RLE)."""
        return {
            "frame_index": int(self.frame_index),
            "ok": bool(self.ok),
            "reason": self.reason,
            "score": float(self.score),
            "box_xyxy": (
                None
                if self.box_xyxy is None
                else [float(value) for value in self.box_xyxy]
            ),
            "mask": None if self.mask is None else mask_to_json_rle(self.mask),
        }

    @staticmethod
    def from_cache_dict(payload: Dict[str, Any]) -> "ProbeResult":
        """Rebuild a result written by :meth:`to_cache_dict`."""
        rle = payload.get("mask")
        box = payload.get("box_xyxy")
        return ProbeResult(
            frame_index=int(payload.get("frame_index", 0)),
            ok=bool(payload.get("ok", False)),
            reason=str(payload.get("reason", "unknown")),
            mask=None if rle is None else json_rle_to_mask(rle),
            score=float(payload.get("score", 0.0)),
            box_xyxy=None if box is None else tuple(float(v) for v in box),
        )


# --------------------------------------------------------------------------
# Pure helpers (unit-testable without a GPU)
# --------------------------------------------------------------------------


def group_queries_by_text(queries: Sequence[Query]) -> Dict[str, List[int]]:
    """Group query indices by text prompt, preserving first-seen order.

    The image backend has to switch text prompts one at a time (setting a new
    text prompt replaces the previous one), so it processes one group per text.
    """
    groups: Dict[str, List[int]] = {}
    for index, (text, _box) in enumerate(queries):
        groups.setdefault(str(text), []).append(index)
    return groups


def pick_best_object(
    scores: Sequence[float],
    boxes: Sequence[Sequence[float]],
    target_box: Sequence[float],
) -> Optional[int]:
    """Index of the predicted object that best matches ``target_box``.

    Highest box IoU wins. When nothing overlaps the target box at all, fall back
    to the highest score so a badly-localised but confident detection is still
    usable (the anchor policy will reject it later if it is wrong).
    """
    if not scores or not boxes:
        return None
    count = min(len(scores), len(boxes))
    if count == 0:
        return None
    ious = [box_iou(boxes[i], target_box) for i in range(count)]
    best_iou = max(ious)
    if best_iou > 0.0:
        return max(
            range(count),
            key=lambda i: (ious[i], float(scores[i]), -i),
        )
    return max(range(count), key=lambda i: (float(scores[i]), -i))


def pick_best_mask(
    masks: Sequence[np.ndarray], target_box: Sequence[float]
) -> Optional[int]:
    """Index of the mask with the highest IoU against ``target_box``."""
    if masks is None or len(masks) == 0:
        return None
    ious = [mask_box_iou(masks[i], target_box) for i in range(len(masks))]
    return max(range(len(ious)), key=lambda i: (ious[i], -i))


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor (CPU or CUDA) or array-like to a numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _image_size(image) -> Tuple[int, int]:
    """``(height, width)`` of a PIL image, numpy array or torch tensor.

    Ambiguity is resolved the way the two ecosystems lay images out: a 3-D array
    whose first axis looks like a channel count is treated as channel-first
    (torch), everything else as height-first (numpy/PIL).
    """
    if hasattr(image, "width") and hasattr(image, "height"):
        return int(image.height), int(image.width)
    shape = getattr(image, "shape", None)
    if shape is None:
        raise ValueError("cannot determine image size")
    dims = [int(value) for value in shape]
    if len(dims) == 2:
        return dims[0], dims[1]
    if len(dims) == 3:
        if dims[0] in (1, 3, 4) and dims[-1] not in (1, 3, 4):
            return dims[1], dims[2]  # channel-first (torch)
        return dims[0], dims[1]  # height-first (numpy/PIL)
    raise ValueError(f"cannot determine image size from shape {tuple(dims)}")


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class ImageModelProbe:
    """Probe keyframes with the SAM3 *image* model via ``Sam3Processor``.

    The model is built lazily on first use so a worker that never probes (for
    example when ``--anchors`` is off) never pays the VRAM cost.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        bpe_path: Optional[str] = None,
        device: str = "cuda",
        resolution: int = 1008,
        confidence_threshold: float = 0.5,
        keep_logits: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.device = device
        self.resolution = int(resolution)
        self.confidence_threshold = float(confidence_threshold)
        self.keep_logits = bool(keep_logits)
        self._processor = None

    def _ensure_processor(self):
        if self._processor is None:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            model = build_sam3_image_model(
                bpe_path=self.bpe_path,
                device=self.device,
                checkpoint_path=self.checkpoint_path,
            )
            self._processor = Sam3Processor(
                model,
                resolution=self.resolution,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
        return self._processor

    def probe_image(
        self,
        image,
        queries: Sequence[Query],
        frame_index: int = 0,
    ) -> List[ProbeResult]:
        """Probe ``queries`` on one image. One result per query, in order."""
        results = [
            ProbeResult(frame_index=frame_index, ok=False, reason="not_probed")
            for _ in queries
        ]
        if not queries:
            return results

        processor = self._ensure_processor()
        state = processor.set_image(image)
        height = int(state["original_height"])
        width = int(state["original_width"])

        for text, indices in group_queries_by_text(queries).items():
            # A new text prompt replaces the previous one, and the box prompts
            # accumulate per text group, so reset before each group.
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(text, state)

            for index in indices:
                target_box = queries[index][1]
                state = processor.add_geometric_prompt(
                    box_xyxy_to_cxcywh_norm(target_box, width, height),
                    True,
                    state,
                )

                masks = state.get("masks")
                boxes = state.get("boxes")
                scores = state.get("scores")
                if masks is None or len(masks) == 0 or boxes is None:
                    results[index] = ProbeResult(frame_index, False, "no_object")
                    continue

                box_list = [_to_numpy(box).reshape(-1) for box in boxes]
                score_list = [float(value) for value in _to_numpy(scores).reshape(-1)]
                best = pick_best_object(score_list, box_list, target_box)
                if best is None:
                    results[index] = ProbeResult(frame_index, False, "no_object")
                    continue

                result = ProbeResult(
                    frame_index=frame_index,
                    ok=True,
                    reason="ok",
                    mask=as_bool_mask(_to_numpy(masks[best])),
                    score=score_list[best] if best < len(score_list) else 0.0,
                    box_xyxy=tuple(float(v) for v in box_list[best]),
                )
                if self.keep_logits:
                    logits = state.get("masks_logits")
                    if logits is not None and best < len(logits):
                        result.logits = _to_numpy(logits[best]).astype(np.float16)
                results[index] = result

        return results

    def close(self) -> None:
        """Release the model and its cached blocks."""
        self._processor = None
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


class VideoModelProbe:
    """Probe keyframes with a one-frame session on an existing video predictor.

    Costs no extra VRAM (it reuses the worker's model) but one session per
    keyframe. The object is chosen by mask-vs-prompt-box IoU, matching how
    ``1_sam3_inference.py`` already selects the tracked object.
    """

    def __init__(self, predictor, keep_logits: bool = False) -> None:
        self.predictor = predictor
        self.keep_logits = bool(keep_logits)

    def probe_image(
        self,
        image,
        queries: Sequence[Query],
        frame_index: int = 0,
    ) -> List[ProbeResult]:
        results = [
            ProbeResult(frame_index=frame_index, ok=False, reason="not_probed")
            for _ in queries
        ]
        if not queries:
            return results

        height, width = _image_size(image)
        response = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=[image])
        )
        session_id = response["session_id"]
        try:
            for index, (text, target_box) in enumerate(queries):
                # add_prompt starts a new semantic state, so one prompt at a time.
                self.predictor.handle_request(
                    request=dict(type="reset_session", session_id=session_id)
                )
                prompt_response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=text,
                        bounding_boxes=[
                            box_xyxy_to_xywh_norm(target_box, width, height)
                        ],
                        bounding_box_labels=[1],
                    )
                )
                outputs = (prompt_response or {}).get("outputs") or {}
                masks = outputs.get("out_binary_masks")
                if masks is None or len(masks) == 0:
                    results[index] = ProbeResult(frame_index, False, "no_object")
                    continue

                best = pick_best_mask(masks, target_box)
                if best is None:
                    results[index] = ProbeResult(frame_index, False, "no_object")
                    continue

                probs = outputs.get("out_probs")
                score = (
                    float(probs[best])
                    if probs is not None and best < len(probs)
                    else 0.0
                )
                results[index] = ProbeResult(
                    frame_index=frame_index,
                    ok=True,
                    reason="ok",
                    mask=as_bool_mask(_to_numpy(masks[best])),
                    score=score,
                    box_xyxy=tuple(float(v) for v in target_box),
                )
        finally:
            try:
                self.predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass

        return results

    def close(self) -> None:
        """No-op: the underlying video predictor is owned by the caller."""
        return None


def build_probe(kind: str = "image", **kwargs):
    """Build a probe backend by name (``"image"`` or ``"video"``)."""
    kind = str(kind).lower()
    if kind == "image":
        return ImageModelProbe(**kwargs)
    if kind == "video":
        predictor = kwargs.pop("predictor", None)
        if predictor is None:
            raise ValueError("the 'video' probe requires a predictor")
        return VideoModelProbe(predictor, **kwargs)
    raise ValueError(f"unknown probe kind {kind!r}; expected 'image' or 'video'")

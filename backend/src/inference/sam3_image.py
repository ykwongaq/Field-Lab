"""Turning a prompt on one frame into a mask.

Two code paths, chosen by what the user is asking for — not by convenience:

* **point / box** → the instance-interactivity predictor (`Sam3Image.predict_inst`).
  These mean *this object*, so they must not go through the concept detector,
  which answers *everything like this* and would hand back a union of instances
  for a single click.
* **text** → the concept detector (`Sam3Processor.set_text_prompt`). A noun phrase
  means *this class*, and the union of its matches is exactly the semantic mask
  the reviewer wants. The individual instances come back too, so an instance-mode
  project can split them into separate objects instead of one blob.

The vision embeddings for a frame are cached, so a click-by-click refinement
re-runs only the decoder. The cache hands out a *fresh* state dict per request
because the detector writes its language features into `backbone_out`; sharing
that dict would leak one prompt's text into the next prompt's result.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.errors import InvalidRequest
from src.core.sessions import Session
from src.domain.prompts import (
    KIND_TEXT,
    InstanceMask,
    SegmentPrompt,
    SegmentResult,
    union_of,
)
from src.inference.frames import load_frame
from src.inference.models import ModelManager, Sam3Config

#: How many frames keep their vision embeddings resident.
CACHE_SIZE = 2


@dataclass
class _BaseState:
    """The cached, prompt-free state of one frame."""

    state: Dict[str, Any]
    height: int
    width: int

    def fresh(self) -> Dict[str, Any]:
        """A copy safe to write prompts into, sharing the (expensive) features."""
        return {
            "original_height": self.height,
            "original_width": self.width,
            "backbone_out": dict(self.state["backbone_out"]),
        }


class Sam3ImageService:
    """Lazily loaded SAM 3 image model, with a small embedding cache."""

    def __init__(
        self,
        config: Optional[Sam3Config] = None,
        manager: Optional[ModelManager] = None,
        *,
        cache_size: int = CACHE_SIZE,
    ) -> None:
        self._manager = manager or ModelManager(config)
        self._cache_size = max(1, cache_size)
        self._cache: "OrderedDict[Tuple[str, int], _BaseState]" = OrderedDict()
        self._lock = threading.RLock()

    # ── availability ────────────────────────────────────────────────────

    @staticmethod
    def installed() -> bool:
        return ModelManager.installed()

    def status(self) -> Dict[str, Any]:
        """Model availability, for the reviewer's status chip."""
        status = self._manager.status()
        status["cache_entries"] = len(self._cache)
        status["point_prompts"] = bool(
            getattr(self._manager.config, "enable_inst_interactivity", True)
        )
        return status

    def warmup(self) -> None:
        """Load the model now (used by the lifespan hook when eager loading is on)."""
        self._manager.warmup()

    def forget(self, session_id: Optional[str] = None) -> None:
        """Drop cached embeddings for a session (or all of them)."""
        with self._lock:
            if session_id is None:
                self._cache.clear()
                return
            for key in [key for key in self._cache if key[0] == session_id]:
                self._cache.pop(key, None)

    # ── segmentation ────────────────────────────────────────────────────

    def segment(
        self, session: Session, frame_index: int, prompt: SegmentPrompt
    ) -> SegmentResult:
        """Apply one prompt to one frame of a session."""
        image = load_frame(session, frame_index)
        width, height = image.size
        prompt.validate(width, height)

        with self._lock, self._manager.inference_context():
            model, processor = self._manager.image()
            base, reused, encoder_ms = self._base_state(
                processor, session.id, frame_index, image, height, width
            )
            started = time.perf_counter()
            if prompt.kind == KIND_TEXT:
                result = self._text_prompt(processor, base, prompt)
            else:
                result = self._object_prompt(model, processor, base, prompt)
            decoder_ms = (time.perf_counter() - started) * 1000.0

        result.height = height
        result.width = width
        result.kind = prompt.kind
        result.prompt = prompt.describe()
        result.encoder_ms = encoder_ms
        result.decoder_ms = decoder_ms
        result.embedding_reused = reused
        return result

    # ── internals ───────────────────────────────────────────────────────

    def _base_state(
        self,
        processor: Any,
        session_id: str,
        frame_index: int,
        image: Any,
        height: int,
        width: int,
    ) -> Tuple[_BaseState, bool, float]:
        """The frame's embeddings, running the vision backbone only when needed."""
        key = (session_id, frame_index)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached, True, 0.0

        started = time.perf_counter()
        state = processor.set_image(image)
        encoder_ms = (time.perf_counter() - started) * 1000.0
        base = _BaseState(state=state, height=height, width=width)
        self._cache[key] = base
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return base, False, encoder_ms

    def _text_prompt(
        self, processor: Any, base: _BaseState, prompt: SegmentPrompt
    ) -> SegmentResult:
        """A concept prompt: every match, plus each match as its own instance."""
        state = processor.set_text_prompt(prompt.text or "", base.fresh())
        masks = state.get("masks")
        scores = state.get("scores")
        instances = _instances_from_batch(masks, scores, base.height, base.width)
        union = (
            union_of([item.mask for item in instances], (base.height, base.width))
            if instances
            else np.zeros((base.height, base.width), dtype=bool)
        )
        return SegmentResult(
            mask=union,
            instances=instances,
            height=base.height,
            width=base.width,
            kind=prompt.kind,
        )

    def _object_prompt(
        self, model: Any, processor: Any, base: _BaseState, prompt: SegmentPrompt
    ) -> SegmentResult:
        """A point and/or box prompt: one object, with SAM's mask candidates."""
        predictor = getattr(model, "inst_interactive_predictor", None)
        if predictor is None:
            # Instance interactivity was switched off at load time; a box can still
            # be answered by the detector, but a click cannot (there is nothing to
            # click *at* without the interactive decoder).
            if prompt.boxes and not prompt.points:
                return self._box_via_detector(processor, base, prompt)
            raise InvalidRequest(
                "Point prompts need SAM 3's instance-interactivity predictor, which is "
                "switched off (`sam3.enable_inst_interactivity` is false). Use a text "
                "or box prompt, or turn it back on."
            )

        kwargs: Dict[str, Any] = {"multimask_output": True}
        if prompt.points:
            # Pixel coordinates: the predictor normalises them against the frame.
            kwargs["point_coords"] = np.array(
                [point.as_xy() for point in prompt.points], dtype=np.float32
            )
            kwargs["point_labels"] = np.array(
                [point.label for point in prompt.points], dtype=np.int32
            )
        box = prompt.primary_box
        if box is not None:
            kwargs["box"] = np.array([box.as_xyxy()], dtype=np.float32)

        masks, scores, _logits = model.predict_inst(base.state, **kwargs)
        instances = _instances_from_batch(masks, scores, base.height, base.width)
        if not instances:
            return SegmentResult(
                mask=np.zeros((base.height, base.width), dtype=bool),
                instances=[],
                height=base.height,
                width=base.width,
                kind=prompt.kind,
            )
        best = max(instances, key=lambda item: item.score)
        return SegmentResult(
            mask=best.mask.copy(),
            instances=instances,
            height=base.height,
            width=base.width,
            kind=prompt.kind,
        )

    def _box_via_detector(
        self, processor: Any, base: _BaseState, prompt: SegmentPrompt
    ) -> SegmentResult:
        """Degraded box prompting: the detector's answer inside the box."""
        box = prompt.primary_box
        assert box is not None  # the caller checked
        state = base.fresh()
        label = bool(box.label)
        state = processor.add_geometric_prompt(
            box.to_cxcywh_normalized(base.width, base.height), label, state
        )
        instances = _instances_from_batch(
            state.get("masks"), state.get("scores"), base.height, base.width
        )
        union = (
            union_of([item.mask for item in instances], (base.height, base.width))
            if instances
            else np.zeros((base.height, base.width), dtype=bool)
        )
        return SegmentResult(
            mask=union,
            instances=instances,
            height=base.height,
            width=base.width,
            kind=prompt.kind,
        )


def _instances_from_batch(
    masks: Any, scores: Any, height: int, width: int
) -> List[InstanceMask]:
    """Normalise whatever shape the models return into `InstanceMask` objects.

    Accepts `(N, H, W)`, `(N, 1, H, W)` and a single `(H, W)` mask, with scores
    either per instance or missing entirely.
    """
    if masks is None:
        return []
    array = np.asarray(masks)
    if array.size == 0:
        return []
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 4:
        array = array[:, 0]
    if array.ndim != 3:
        raise InvalidRequest(
            f"A segmentation model returned an unexpected mask shape {array.shape}."
        )

    if scores is None:
        score_list = [1.0] * array.shape[0]
    else:
        flat = np.asarray(scores).reshape(-1)
        score_list = [float(value) for value in flat[: array.shape[0]]]

    instances: List[InstanceMask] = []
    for index in range(array.shape[0]):
        binary = np.asarray(array[index]) > 0
        if binary.shape != (height, width):
            continue
        if not binary.any():
            continue
        score = score_list[index] if index < len(score_list) else 1.0
        instances.append(InstanceMask(mask=binary, score=score))
    return instances

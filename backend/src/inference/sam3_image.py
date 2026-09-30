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

from src.core.config import (
    DEFAULT_SAM3_IMAGE_CACHE_CLIENTS,
    DEFAULT_SAM3_IMAGE_CACHE_SIZE,
)
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

#: How many frames keep their vision embeddings resident when the caller does
#: not say. `sam3.image_cache_size` in `core.config` is the configured value;
#: this is its default.
CACHE_SIZE = DEFAULT_SAM3_IMAGE_CACHE_SIZE

#: How many reviewers keep such a cache when the caller does not say.
CACHE_CLIENTS = DEFAULT_SAM3_IMAGE_CACHE_CLIENTS


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
    """Lazily loaded SAM 3 image model, with a small per-reviewer embedding cache."""

    def __init__(
        self,
        config: Optional[Sam3Config] = None,
        manager: Optional[ModelManager] = None,
        *,
        cache_size: int = CACHE_SIZE,
        cache_clients: int = CACHE_CLIENTS,
    ) -> None:
        self._manager = manager or ModelManager(config)
        self._cache_size = max(1, cache_size)
        self._cache_clients = max(1, cache_clients)
        #: reviewer -> (session, frame) -> embeddings. Two levels because the
        #: budget has to be split *fairly*: one flat LRU lets a reviewer's clicks
        #: evict another reviewer's frames, and every eviction costs a full
        #: vision-backbone pass. Frames are LRU within a reviewer, and reviewers
        #: are LRU across each other, so an idle one gives up its whole cache
        #: before an active one loses a frame.
        self._cache: "OrderedDict[str, OrderedDict[Tuple[str, int], _BaseState]]" = (
            OrderedDict()
        )
        self._lock = threading.RLock()

    # ── availability ────────────────────────────────────────────────────

    @staticmethod
    def installed() -> bool:
        return ModelManager.installed()

    def status(self) -> Dict[str, Any]:
        """Model availability, for the reviewer's status chip."""
        status = self._manager.status()
        # Summed under the lock: a request thread can be adding a reviewer's
        # first entry while a status poll walks the mapping.
        with self._lock:
            status["cache_entries"] = sum(
                len(frames) for frames in self._cache.values()
            )
        status["point_prompts"] = bool(
            getattr(self._manager.config, "enable_inst_interactivity", True)
        )
        return status

    def warmup(self) -> None:
        """Load the model now (used by the lifespan hook when eager loading is on)."""
        self._manager.warmup()

    # ── segmentation ────────────────────────────────────────────────────

    def segment(
        self, session: Session, frame_index: int, prompt: SegmentPrompt
    ) -> SegmentResult:
        """Apply one prompt to one frame of a session."""
        image = load_frame(session, frame_index)
        width, height = image.size
        prompt.validate(width, height)

        # The gate is taken first: a prompt waits for at most one propagation
        # window, and nothing can evict the weights this call is about to use.
        with (
            self._manager.gate.interactive(),
            self._lock,
            self._manager.inference_context(),
        ):
            model, processor = self._manager.image()
            base, reused, encoder_ms = self._base_state(
                processor,
                session.owner,
                session.id,
                frame_index,
                image,
                height,
                width,
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
        owner: str,
        session_id: str,
        frame_index: int,
        image: Any,
        height: int,
        width: int,
    ) -> Tuple[_BaseState, bool, float]:
        """The frame's embeddings, running the vision backbone only when needed.

        The reviewer is the cache's outer key, so a click on someone else's frame
        cannot cost this reviewer a re-encode. `session_id` is part of the inner
        key rather than the outer one because one reviewer may have several
        clips open, and two clips can share a frame index.
        """
        frames = self._cache.get(owner)
        if frames is not None:
            cached = frames.get((session_id, frame_index))
            if cached is not None:
                self._cache.move_to_end(owner)
                frames.move_to_end((session_id, frame_index))
                return cached, True, 0.0

        started = time.perf_counter()
        state = processor.set_image(image)
        encoder_ms = (time.perf_counter() - started) * 1000.0
        base = _BaseState(state=state, height=height, width=width)
        if frames is None:
            frames = OrderedDict()
            self._cache[owner] = frames
        frames[(session_id, frame_index)] = base
        self._cache.move_to_end(owner)
        self._evict(owner)
        return base, False, encoder_ms

    def _evict(self, owner: str) -> None:
        """Hold the cache to `cache_clients` reviewers × `cache_size` frames."""
        frames = self._cache[owner]
        while len(frames) > self._cache_size:
            frames.popitem(last=False)
        while len(self._cache) > self._cache_clients:
            # `owner` was just moved to the end, so this drops another reviewer —
            # the one that has not clicked for longest — and never the caller.
            self._cache.popitem(last=False)

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
        if box is None:  # the caller checked, so this cannot happen
            raise InvalidRequest("A box prompt reached the detector with no box.")
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


def _to_numpy(value: Any) -> Any:
    """Bring a model output onto the host so numpy can read it.

    The models hand back torch tensors that are still on the GPU, and
    `np.asarray` on a CUDA tensor refuses rather than copying: "can't convert
    cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to
    host memory first." Torch is duck-typed rather than imported because this
    module has to stay importable without it (see the module docstring), and a
    numpy array or a plain list has no `detach` and falls straight through.
    """
    detach = getattr(value, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    return value


def _instances_from_batch(
    masks: Any, scores: Any, height: int, width: int
) -> List[InstanceMask]:
    """Normalise whatever shape the models return into `InstanceMask` objects.

    Accepts `(N, H, W)`, `(N, 1, H, W)` and a single `(H, W)` mask, with scores
    either per instance or missing entirely. This is the boundary where a model
    result stops being torch-shaped and becomes numpy-shaped, which is why the
    tensors are moved to the host here rather than anywhere downstream.
    """
    if masks is None:
        return []
    array = np.asarray(_to_numpy(masks))
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
        flat = np.asarray(_to_numpy(scores)).reshape(-1)
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

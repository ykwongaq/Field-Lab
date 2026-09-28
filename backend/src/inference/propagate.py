"""Mask propagation over a window of frames.

A mask given on one anchor frame is tracked forward and/or backward through the
neighbouring frames. The browser uploads exactly the frames of the window, so
request size stays bounded and long stretches are done in several runs.

Configuration
-------------
PROPAGATE_MAX_FRAMES   Maximum frames per request incl. the anchor (default 120).
SAM3_MODEL / SAM3_DEVICE / SAM3_DTYPE
                       Shared with `inference/sam3.py`.
SAM3_TRACKER_MODEL     Checkpoint for the tracker when it differs from
                       SAM3_MODEL (default: same as SAM3_MODEL; the official
                       ``facebook/sam3`` repo contains the tracker weights).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.config import Settings
from src.core.errors import InvalidRequest, Unavailable
from src.domain.images import decode_image_rgb
from src.inference.sam3 import Sam3Config

DEFAULT_MAX_FRAMES = 120


class PropagateUnavailable(Unavailable):
    """The tracker cannot be used (not installed, load failure)."""


class PropagateError(InvalidRequest):
    """The request is malformed (bad range, mask/frame size mismatch, ...)."""


@dataclass(frozen=True)
class FrameInput:
    """One frame of the window: its index in the clip and its image bytes."""

    index: int
    data: bytes


@dataclass
class PropagatedMask:
    """The tracker's mask for one frame of the window."""

    frame_index: int
    mask: np.ndarray  # bool (H, W)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class PropagateResult:
    masks: List[PropagatedMask]
    backend: str
    model: str
    device: str
    height: int
    width: int
    elapsed_ms: float


@dataclass
class PropagateConfig:
    """Request limits and the tracker checkpoint to use."""

    max_frames: int = DEFAULT_MAX_FRAMES
    tracker_model: Optional[str] = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "PropagateConfig":
        return cls(
            max_frames=settings.propagate_max_frames,
            tracker_model=settings.sam3_tracker_model,
        )


def validate_window(
    frames: Sequence[FrameInput], anchor: int, max_frames: int
) -> Tuple[List[FrameInput], int]:
    """Sort the window, check it is contiguous and return `(frames, anchor_pos)`.

    `anchor` is the clip frame index of the anchor; the returned position is its
    offset inside the sorted window.
    """
    if not frames:
        raise PropagateError("The window contains no frame.")
    if len(frames) > max_frames:
        raise PropagateError(
            f"The window has {len(frames)} frames; the limit is {max_frames} "
            "(PROPAGATE_MAX_FRAMES). Propagate in shorter runs."
        )
    ordered = sorted(frames, key=lambda f: f.index)
    indices = [f.index for f in ordered]
    if len(set(indices)) != len(indices):
        raise PropagateError("The window lists the same frame twice.")
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise PropagateError("The window must be a contiguous range of frames.")
    if anchor not in indices:
        raise PropagateError(f"Anchor frame {anchor} is not part of the window.")
    for frame in ordered:
        if not frame.data:
            raise PropagateError(f"Frame {frame.index} is empty.")
    return ordered, indices.index(anchor)


class Sam3TrackerPropagator:
    """Lazy, lock-protected wrapper around Transformers' ``Sam3TrackerVideoModel``."""

    name = "sam3"

    def __init__(
        self,
        config: Optional[Sam3Config] = None,
        tracker_model: Optional[str] = None,
    ) -> None:
        self.config = config or Sam3Config()
        self.model_id = tracker_model or self.config.hf_model
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None
        self._device: Optional[str] = None
        self._error: Optional[str] = None

    @staticmethod
    def installed() -> bool:
        try:
            import transformers

            return hasattr(transformers, "Sam3TrackerVideoModel")
        except Exception:
            return False

    def status(self) -> Dict[str, Any]:
        if not self.installed():
            error: Optional[str] = (
                "transformers has no Sam3TrackerVideoModel; install "
                "transformers>=5.0 (pip install -r backend/requirements.txt)."
            )
        else:
            error = self._error
        return {
            "available": error is None,
            "loaded": self._model is not None,
            "model": self.model_id,
            "device": self._device or self.config.device,
            "error": error,
        }

    def _resolve_device_name(self) -> str:
        import torch

        device = self.config.device
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _dtype(self, device: str) -> Any:
        import torch

        if self.config.dtype == "auto":
            return torch.bfloat16 if device.startswith("cuda") else torch.float32
        return getattr(torch, self.config.dtype)

    def warmup(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def _ensure_loaded(self) -> Tuple[Any, Any]:
        """Must be called with `self._lock` held."""
        if self._model is not None:
            return self._model, self._processor
        if self._error is not None:
            raise PropagateUnavailable(self._error)
        try:
            import torch
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

            device = self._resolve_device_name()
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise PropagateUnavailable(
                    f"SAM3_DEVICE={device!r} but no CUDA device is visible to torch."
                )
            model = Sam3TrackerVideoModel.from_pretrained(
                self.model_id, dtype=self._dtype(device)
            )
            model = model.to(device).eval()
            processor = Sam3TrackerVideoProcessor.from_pretrained(self.model_id)
            self._model, self._processor, self._device = model, processor, device
            return model, processor
        except PropagateUnavailable as exc:
            self._error = str(exc)
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if "gated" in message.lower() or "401" in message:
                message += (
                    " — request access at https://huggingface.co/facebook/sam3 "
                    "and run `hf auth login`."
                )
            self._error = message
            raise PropagateUnavailable(message) from exc

    def propagate(
        self,
        frames: Sequence[FrameInput],
        anchor: int,
        mask: np.ndarray,
        *,
        backward: int,
        forward: int,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ) -> PropagateResult:
        """Track `mask` (bool HxW on frame `anchor`) over the window."""
        window, pos = validate_window(frames, anchor, max_frames)
        started = time.perf_counter()

        images = [decode_image_rgb(frame.data) for frame in window]
        height, width = images[0].shape[:2]
        if any(image.shape[:2] != (height, width) for image in images):
            raise PropagateError("All frames of the window must have the same size.")
        if mask.shape != (height, width):
            raise PropagateError(
                f"The mask is {mask.shape[1]}x{mask.shape[0]} but the frames are "
                f"{width}x{height}."
            )

        with self._lock:
            model, processor = self._ensure_loaded()
            import torch

            device = self._device or "cpu"
            with torch.inference_mode():
                session = processor.init_video_session(
                    video=images,
                    inference_device=device,
                    video_storage_device="cpu",
                    dtype=self._dtype(device),
                )
                processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=pos,
                    obj_ids=1,
                    input_masks=mask.astype(np.float32),
                )

                model(inference_session=session, frame_idx=pos)
                results: List[PropagatedMask] = []
                for reverse, count in ((False, forward), (True, backward)):
                    if count <= 0:
                        continue
                    for output in model.propagate_in_video_iterator(
                        session,
                        start_frame_idx=pos,
                        max_frame_num_to_track=count,
                        reverse=reverse,
                    ):
                        if output.frame_idx == pos:
                            continue
                        binary = processor.post_process_masks(
                            [output.pred_masks],
                            original_sizes=[[height, width]],
                            binarize=True,
                        )[0]
                        predicted = binary[0, 0].to("cpu").numpy().astype(bool)
                        results.append(
                            PropagatedMask(window[output.frame_idx].index, predicted)
                        )

        results.sort(key=lambda r: r.frame_index)
        return PropagateResult(
            masks=results,
            backend=self.name,
            model=self.model_id,
            device=self._device or "?",
            height=height,
            width=width,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


class PropagateService:
    """Validates a propagation request and runs the SAM 3 tracker."""

    def __init__(
        self,
        config: Optional[PropagateConfig] = None,
        tracker_config: Optional[Sam3Config] = None,
    ) -> None:
        self.config = config or PropagateConfig()
        self.sam3 = Sam3TrackerPropagator(
            config=tracker_config or Sam3Config(),
            tracker_model=self.config.tracker_model,
        )

    def status(self) -> Dict[str, Any]:
        return {"sam3": self.sam3.status(), "max_frames": self.config.max_frames}

    def propagate(
        self,
        backend: str,
        frames: Sequence[FrameInput],
        anchor: int,
        mask: np.ndarray,
        *,
        backward: int,
        forward: int,
    ) -> PropagateResult:
        """Run the requested tracker. Only ``"sam3"`` is available."""
        if backend != self.sam3.name:
            raise PropagateError(
                f"Unknown propagation backend {backend!r}; only "
                f"{self.sam3.name!r} is available."
            )
        if backward < 0 or forward < 0:
            raise PropagateError("`backward` and `forward` must be >= 0.")
        if backward == 0 and forward == 0:
            raise PropagateError("Nothing to propagate: both ranges are 0.")
        return self.sam3.propagate(
            frames,
            anchor,
            mask,
            backward=backward,
            forward=forward,
            max_frames=self.config.max_frames,
        )

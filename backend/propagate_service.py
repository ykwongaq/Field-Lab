"""
Configuration
-------------------------------------
PROPAGATE_MAX_FRAMES   Maximum frames per request incl. the anchor (default 120).
SAM2_MODEL / SAM2_CHECKPOINT / SAM2_CONFIG / SAM2_DEVICE
                       Shared with ``sam2_service.py``; the video predictor is
                       built from the same checkpoint.
SAM3_MODEL / SAM3_DEVICE / SAM3_DTYPE
                       Shared with ``sam3_service.py``.
SAM3_TRACKER_MODEL     Checkpoint for the tracker when it differs from
                       SAM3_MODEL (default: same as SAM3_MODEL; the official
                       ``facebook/sam3`` repo contains the tracker weights).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sam2_service import Sam2Config
from sam3_service import Sam3Config

DEFAULT_MAX_FRAMES = 120


class PropagateUnavailable(RuntimeError):
    """The requested tracker cannot be used (not installed, load failure)."""


class PropagateError(ValueError):
    """The request is malformed (bad range, mask size mismatch, ...)."""


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

    def rle(self) -> Dict[str, Any]:
        """pycocotools compressed RLE, `counts` as a UTF-8 string."""
        from pycocotools import mask as mask_utils

        encoded = mask_utils.encode(np.asfortranarray(self.mask.astype(np.uint8)))
        counts = encoded["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        height, width = self.mask.shape
        return {"size": [height, width], "counts": counts}


@dataclass
class PropagateResult:
    masks: List[PropagatedMask]
    backend: str  # "sam2" | "sam3"
    model: str
    device: str
    height: int
    width: int
    elapsed_ms: float


@dataclass
class PropagateConfig:
    max_frames: int = field(
        default_factory=lambda: int(
            os.environ.get("PROPAGATE_MAX_FRAMES", str(DEFAULT_MAX_FRAMES))
        )
    )
    sam3_tracker_model: Optional[str] = field(
        default_factory=lambda: os.environ.get("SAM3_TRACKER_MODEL") or None
    )



def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """pycocotools RLE (``{"size": [h, w], "counts": str}``) -> bool (H, W)."""
    from pycocotools import mask as mask_utils

    try:
        size = [int(rle["size"][0]), int(rle["size"][1])]
        counts = rle["counts"]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise PropagateError("`mask` must be {size: [h, w], counts: str}.") from exc
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    if isinstance(counts, list):  # uncompressed counts
        encoded = mask_utils.frPyObjects({"size": size, "counts": counts}, *size)
    else:
        encoded = {"size": size, "counts": counts}
    return mask_utils.decode(encoded).astype(bool)


def decode_image(image_bytes: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB uint8 array (H, W, 3)."""
    import cv2

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise PropagateError("A frame could not be decoded as an image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _is_jpeg(data: bytes) -> bool:
    return data[:3] == b"\xff\xd8\xff"


def validate_window(
    frames: Sequence[FrameInput], anchor: int, max_frames: int
) -> Tuple[List[FrameInput], int]:
    """Sort the window, check it is contiguous and return `(frames, anchor_pos)`.

    `anchor` is the clip frame index of the anchor; the returned position is
    its offset inside the sorted window.
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


# SAM 2


class Sam2Propagator:

    name = "sam2"

    def __init__(self, config: Optional[Sam2Config] = None) -> None:
        self.config = config or Sam2Config()
        self._lock = threading.Lock()
        self._predictor: Any = None
        self._device: Optional[str] = None
        self._error: Optional[str] = None

    @staticmethod
    def installed() -> bool:
        try:
            import sam2  # noqa: F401

            return True
        except Exception:
            return False

    def status(self) -> Dict[str, Any]:
        if not self.installed():
            error = "The `sam2` package is not installed (pip install -r requirements-sam2.txt)."
        else:
            error = self._error
        return {
            "available": error is None,
            "loaded": self._predictor is not None,
            "model": self.config.describe_model(),
            "device": self._device or self.config.device,
            "error": error,
        }

    def _resolve_device_name(self) -> str:
        import torch

        device = self.config.device
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def warmup(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def _ensure_loaded(self) -> Any:
        """Must be called with `self._lock` held."""
        if self._predictor is not None:
            return self._predictor
        if self._error is not None:
            raise PropagateUnavailable(self._error)
        try:
            import torch

            device = self._resolve_device_name()
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise PropagateUnavailable(
                    f"SAM2_DEVICE={device!r} but no CUDA device is visible to torch."
                )
            if self.config.checkpoint:
                if not self.config.model_cfg:
                    raise PropagateUnavailable(
                        "SAM2_CONFIG must be set together with SAM2_CHECKPOINT."
                    )
                from sam2.build_sam import build_sam2_video_predictor

                predictor = build_sam2_video_predictor(
                    self.config.model_cfg, self.config.checkpoint, device=device
                )
            else:
                from sam2.sam2_video_predictor import SAM2VideoPredictor

                predictor = SAM2VideoPredictor.from_pretrained(
                    self.config.hf_model, device=device
                )
            self._predictor = predictor
            self._device = device
            return predictor
        except PropagateUnavailable as exc:
            self._error = str(exc)
            raise
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            raise PropagateUnavailable(self._error) from exc

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
        """Track `mask` (bool H×W on frame `anchor`) over the window."""
        window, pos = validate_window(frames, anchor, max_frames)
        started = time.perf_counter()

        with self._lock:
            predictor = self._ensure_loaded()
            import torch

            folder = tempfile.mkdtemp(prefix="vsr-propagate-")
            try:
                height, width = self._write_frames(window, folder)
                if mask.shape != (height, width):
                    raise PropagateError(
                        f"The mask is {mask.shape[1]}x{mask.shape[0]} but the frames are {width}x{height}."
                    )
                results: List[PropagatedMask] = []
                with torch.inference_mode():
                    state = predictor.init_state(
                        video_path=folder,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=False,
                    )
                    predictor.add_new_mask(
                        state, frame_idx=pos, obj_id=1, mask=torch.from_numpy(mask)
                    )
                    for reverse, count in ((False, forward), (True, backward)):
                        if count <= 0:
                            continue
                        for frame_pos, _obj_ids, logits in predictor.propagate_in_video(
                            state,
                            start_frame_idx=pos,
                            max_frame_num_to_track=count,
                            reverse=reverse,
                        ):
                            if frame_pos == pos:
                                continue
                            predicted = (logits[0, 0] > 0.0).cpu().numpy()
                            results.append(
                                PropagatedMask(window[frame_pos].index, predicted)
                            )
                    predictor.reset_state(state)
            finally:
                shutil.rmtree(folder, ignore_errors=True)

        results.sort(key=lambda r: r.frame_index)
        return PropagateResult(
            masks=results,
            backend=self.name,
            model=self.config.describe_model(),
            device=self._device or "?",
            height=height,
            width=width,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _write_frames(window: Sequence[FrameInput], folder: str) -> Tuple[int, int]:
        """Write the window as `00000.jpg`, `00001.jpg`, ... and return (H, W)."""
        import cv2

        size: Optional[Tuple[int, int]] = None
        for position, frame in enumerate(window):
            data = frame.data
            image = decode_image(data)
            if size is None:
                size = image.shape[:2]
            elif image.shape[:2] != size:
                raise PropagateError("All frames of the window must have the same size.")
            if not _is_jpeg(data):
                ok, encoded = cv2.imencode(
                    ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                if not ok:
                    raise PropagateError(f"Frame {frame.index} could not be re-encoded as JPEG.")
                data = encoded.tobytes()
            with open(os.path.join(folder, f"{position:05d}.jpg"), "wb") as handle:
                handle.write(data)
        assert size is not None
        return int(size[0]), int(size[1])


# SAM 3 tracker


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
            error = (
                "transformers has no Sam3TrackerVideoModel; install transformers>=5.0 "
                "(pip install -r requirements-sam3.txt)."
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
                message += " — request access at https://huggingface.co/facebook/sam3 and run `hf auth login`."
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
        """Track `mask` (bool H×W on frame `anchor`) over the window."""
        window, pos = validate_window(frames, anchor, max_frames)
        started = time.perf_counter()
        images = [decode_image(frame.data) for frame in window]
        height, width = images[0].shape[:2]
        if any(image.shape[:2] != (height, width) for image in images):
            raise PropagateError("All frames of the window must have the same size.")
        if mask.shape != (height, width):
            raise PropagateError(
                f"The mask is {mask.shape[1]}x{mask.shape[0]} but the frames are {width}x{height}."
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
    """Chooses the tracker for a request and reports both statuses."""

    def __init__(self, config: Optional[PropagateConfig] = None) -> None:
        self.config = config or PropagateConfig()
        self.sam2 = Sam2Propagator()
        self.sam3 = Sam3TrackerPropagator(tracker_model=self.config.sam3_tracker_model)

    def status(self) -> Dict[str, Any]:
        return {
            "sam2": self.sam2.status(),
            "sam3": self.sam3.status(),
            "max_frames": self.config.max_frames,
        }

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
        """Run the requested tracker; ``backend`` is "sam2" or "sam3"."""
        if backend == "sam2":
            tracker: Any = self.sam2
        elif backend == "sam3":
            tracker = self.sam3
        else:
            raise PropagateError(f"Unknown propagation backend {backend!r}.")
        if backward < 0 or forward < 0:
            raise PropagateError("`backward` and `forward` must be >= 0.")
        if backward == 0 and forward == 0:
            raise PropagateError("Nothing to propagate: both ranges are 0.")
        return tracker.propagate(
            frames,
            anchor,
            mask,
            backward=backward,
            forward=forward,
            max_frames=self.config.max_frames,
        )


service = PropagateService()

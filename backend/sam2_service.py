"""SAM 2 image segmentation

Configuration
-------------------------------------
SAM2_MODEL        Hugging Face model id, default "facebook/sam2.1-hiera-small".
                  Used when SAM2_CHECKPOINT is not set (weights are downloaded
                  to the HF cache on first use).
SAM2_CHECKPOINT   Path to a local ``*.pt`` checkpoint. Requires SAM2_CONFIG.
SAM2_CONFIG       Hydra config name for the checkpoint, e.g.
                  "configs/sam2.1/sam2.1_hiera_s.yaml".
SAM2_DEVICE       "cuda", "cuda:1", "cpu" or "auto" (default: cuda if available).
SAM2_EAGER        "1" to load the model at API start-up instead of lazily.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_HF_MODEL = "facebook/sam2.1-hiera-small"

POSITIVE = 1
NEGATIVE = 0


class Sam2Unavailable(RuntimeError):
    """SAM 2 cannot be used."""


class PromptError(ValueError):
    """The click prompt is malformed."""


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


@dataclass
class SegmentResult:
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
        """`(x, y, w, h)` of the mask or `None` for an empty mask."""
        ys, xs = np.nonzero(self.mask)
        if xs.size == 0:
            return None
        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )

    def rle(self) -> Dict[str, Any]:
        """pycocotools compressed RLE, `counts` as a UTF-8 string."""
        from pycocotools import mask as mask_utils

        encoded = mask_utils.encode(np.asfortranarray(self.mask.astype(np.uint8)))
        counts = encoded["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return {"size": [self.height, self.width], "counts": counts}


@dataclass
class Sam2Config:
    hf_model: str = field(
        default_factory=lambda: os.environ.get("SAM2_MODEL", DEFAULT_HF_MODEL)
    )
    checkpoint: Optional[str] = field(
        default_factory=lambda: os.environ.get("SAM2_CHECKPOINT") or None
    )
    model_cfg: Optional[str] = field(
        default_factory=lambda: os.environ.get("SAM2_CONFIG") or None
    )
    device: str = field(default_factory=lambda: os.environ.get("SAM2_DEVICE", "auto"))
    eager: bool = field(default_factory=lambda: os.environ.get("SAM2_EAGER") == "1")

    def describe_model(self) -> str:
        if self.checkpoint:
            return os.path.basename(self.checkpoint)
        return self.hf_model


class Sam2Service:

    def __init__(self, config: Optional[Sam2Config] = None) -> None:
        self.config = config or Sam2Config()
        self._lock = threading.Lock()
        self._predictor: Any = None
        self._device: Optional[str] = None
        self._error: Optional[str] = None
        self._last_image_key: Optional[str] = None
        self._last_image_size: Optional[Tuple[int, int]] = None  # (h, w)

    # status

    @staticmethod
    def installed() -> bool:
        try:
            import sam2  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            return False
        return True

    def status(self) -> Dict[str, Any]:
        installed = self.installed()
        loaded = self._predictor is not None
        error = self._error
        if not installed and error is None:
            error = (
                "SAM 2 is not installed in the backend environment "
                "(pip install -r requirements-sam2.txt)."
            )
        if self.config.checkpoint and not os.path.exists(self.config.checkpoint):
            error = f"SAM2_CHECKPOINT not found: {self.config.checkpoint}"
        if self.config.checkpoint and not self.config.model_cfg:
            error = "SAM2_CONFIG must be set together with SAM2_CHECKPOINT."
        return {
            "available": installed and error is None,
            "loaded": loaded,
            "model": self.config.describe_model(),
            "device": self._device or self._resolve_device_name(),
            "error": error,
        }

    def _resolve_device_name(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    # load model

    def warmup(self) -> None:
        """Load the model; raises `Sam2Unavailable` on failure."""
        with self._lock:
            self._ensure_loaded()

    def _ensure_loaded(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        if self._error is not None:
            raise Sam2Unavailable(self._error)
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = self._resolve_device_name()
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise Sam2Unavailable(
                    f"SAM2_DEVICE={device!r} but no CUDA device is visible to torch."
                )
            if self.config.checkpoint:
                if not self.config.model_cfg:
                    raise Sam2Unavailable(
                        "SAM2_CONFIG must be set together with SAM2_CHECKPOINT."
                    )
                if not os.path.exists(self.config.checkpoint):
                    raise Sam2Unavailable(
                        f"SAM2_CHECKPOINT not found: {self.config.checkpoint}"
                    )
                from sam2.build_sam import build_sam2

                model = build_sam2(
                    self.config.model_cfg, self.config.checkpoint, device=device
                )
                predictor = SAM2ImagePredictor(model)
            else:
                predictor = SAM2ImagePredictor.from_pretrained(
                    self.config.hf_model, device=device
                )
            self._predictor = predictor
            self._device = device
            return predictor
        except Sam2Unavailable as exc:
            self._error = str(exc)
            raise
        except Exception as exc:  # import errors, corrupt checkpoint, OOM, ...
            self._error = f"{type(exc).__name__}: {exc}"
            raise Sam2Unavailable(self._error) from exc

    # inference

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        """JPEG/PNG bytes -> RGB uint8 array (H, W, 3)."""
        import cv2

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            raise PromptError("The uploaded frame could not be decoded as an image.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def segment(
        self,
        image_bytes: bytes,
        points: Sequence[PromptPoint],
        *,
        image_key: Optional[str] = None,
    ) -> SegmentResult:
        """Predict one mask for `points` on the given frame.
        """
        if not points:
            raise PromptError("At least one prompt point is required.")
        key = image_key or hashlib.sha1(image_bytes).hexdigest()

        with self._lock:
            predictor = self._ensure_loaded()
            import torch

            reuse = key == self._last_image_key and self._last_image_size is not None
            encoder_ms = 0.0
            if reuse:
                height, width = self._last_image_size  # type: ignore[misc]
            else:
                image = self._decode_image(image_bytes)
                height, width = image.shape[:2]
                started = time.perf_counter()
                with torch.inference_mode():
                    predictor.set_image(image)
                encoder_ms = (time.perf_counter() - started) * 1000
                self._last_image_key = key
                self._last_image_size = (height, width)

            for point in points:
                point.validate(width, height)

            coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
            labels = np.array([p.label for p in points], dtype=np.int32)
            
            multimask = len(points) == 1
            started = time.perf_counter()
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=multimask,
                )
            decoder_ms = (time.perf_counter() - started) * 1000

        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)
        best = int(np.argmax(scores)) if masks.shape[0] > 1 else 0
        mask = masks[best].astype(bool)
        return SegmentResult(
            mask=mask,
            score=float(scores[best]),
            height=height,
            width=width,
            encoder_ms=encoder_ms,
            decoder_ms=decoder_ms,
            embedding_reused=reuse,
        )


def parse_points(raw: Any) -> List[PromptPoint]:
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


service = Sam2Service()

"""SAM 3

Configuration
-------------------------------------
SAM3_MODEL          Hugging Face model id or local directory, default
                    "facebook/sam3".
SAM3_DEVICE         "cuda", "cuda:1", "cpu" or "auto" (default).
SAM3_DTYPE          "auto" (bfloat16 on CUDA, float32 on CPU), "float32",
                    "bfloat16" or "float16".
SAM3_THRESHOLD      Detection score threshold, default 0.5. Lower it to find
                    more (and noisier) regions.
SAM3_MASK_THRESHOLD Mask binarisation threshold, default 0.5.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sam2_service import NEGATIVE, POSITIVE, PromptError, PromptPoint, SegmentResult

DEFAULT_HF_MODEL = "facebook/sam3"

BoxRefiner = Callable[[PromptPoint], Optional[Tuple[int, int, int, int]]]


class Sam3Unavailable(RuntimeError):
    """SAM 3 cannot be used (not installed, no access to the weights, ...)."""


@dataclass(frozen=True)
class ExemplarBox:
    """One exemplar box in frame pixels, `label` 1 = example, 0 = counter-example."""

    x0: float
    y0: float
    x1: float
    y1: float
    label: int

    def validate(self, width: int, height: int) -> None:
        if self.label not in (POSITIVE, NEGATIVE):
            raise PromptError(f"box label must be 0 or 1, got {self.label!r}")
        if not (0 <= self.x0 < self.x1 <= width and 0 <= self.y0 < self.y1 <= height):
            raise PromptError(
                f"box ({self.x0:.0f}, {self.y0:.0f})-({self.x1:.0f}, {self.y1:.0f}) "
                f"is not inside the {width}x{height} frame"
            )

    def as_list(self) -> List[float]:
        return [float(self.x0), float(self.y0), float(self.x1), float(self.y1)]


@dataclass
class ConceptResult(SegmentResult):
    """Union of every detected instance of the concept, plus per-instance scores."""

    instances: int = 0
    instance_scores: List[float] = field(default_factory=list)
    exemplars: List[ExemplarBox] = field(default_factory=list)


@dataclass
class Sam3Config:
    hf_model: str = field(
        default_factory=lambda: os.environ.get("SAM3_MODEL", DEFAULT_HF_MODEL)
    )
    device: str = field(default_factory=lambda: os.environ.get("SAM3_DEVICE", "auto"))
    dtype: str = field(default_factory=lambda: os.environ.get("SAM3_DTYPE", "auto"))
    threshold: float = field(
        default_factory=lambda: float(os.environ.get("SAM3_THRESHOLD", "0.5"))
    )
    mask_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SAM3_MASK_THRESHOLD", "0.5"))
    )
    exemplar_fraction: float = field(
        default_factory=lambda: float(os.environ.get("SAM3_EXEMPLAR_BOX", "0.06"))
    )
    eager: bool = field(default_factory=lambda: os.environ.get("SAM3_EAGER") == "1")

    def describe_model(self) -> str:
        return os.path.basename(self.hf_model.rstrip("/")) if os.path.isdir(
            self.hf_model
        ) else self.hf_model



ModelLoader = Callable[[Sam3Config, str], Tuple[Any, Any]]


def exemplar_boxes_from_points(
    points: Sequence[PromptPoint],
    width: int,
    height: int,
    *,
    refine: Optional[BoxRefiner] = None,
    fraction: float = 0.06,
) -> List[ExemplarBox]:
    """Turn click prompts into exemplar boxes.
    """
    boxes: List[ExemplarBox] = []
    side = max(8.0, fraction * min(width, height))
    for point in points:
        point.validate(width, height)
        box: Optional[Tuple[int, int, int, int]] = None
        if refine is not None:
            try:
                box = refine(PromptPoint(point.x, point.y, POSITIVE))
            except Exception:
                box = None
        if box is not None and box[2] > 0 and box[3] > 0:
            x, y, w, h = box
            x0, y0, x1, y1 = float(x), float(y), float(x + w), float(y + h)
        else:
            half = side / 2
            x0, y0 = point.x - half, point.y - half
            x1, y1 = point.x + half, point.y + half
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(width), x1), min(float(height), y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        boxes.append(ExemplarBox(x0, y0, x1, y1, point.label))
    return boxes


def _default_loader(config: Sam3Config, device: str) -> Tuple[Any, Any]:
    """Load `Sam3Model` + `Sam3Processor`."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    if config.dtype == "auto":
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    else:
        dtype = getattr(torch, config.dtype)
    model = Sam3Model.from_pretrained(config.hf_model, dtype=dtype)
    model = model.to(device).eval()
    processor = Sam3Processor.from_pretrained(config.hf_model)
    return model, processor


class Sam3Service:

    def __init__(
        self,
        config: Optional[Sam3Config] = None,
        loader: Optional[ModelLoader] = None,
    ) -> None:
        self.config = config or Sam3Config()
        self._loader = loader or _default_loader
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None
        self._device: Optional[str] = None
        self._error: Optional[str] = None
        self._last_image_key: Optional[str] = None
        self._last_image: Any = None  # PIL image (needed to normalise boxes)
        self._last_vision_embeds: Any = None

    # status

    @staticmethod
    def installed() -> bool:
        try:
            import torch
            import transformers

            return hasattr(transformers, "Sam3Model")
        except Exception:
            return False

    def status(self) -> Dict[str, Any]:
        installed = self.installed()
        error = self._error
        if not installed and error is None:
            error = (
                "SAM 3 is not installed in the backend environment "
                "(pip install -r requirements-sam3.txt)."
            )
        return {
            "available": installed and error is None,
            "loaded": self._model is not None,
            "model": self.config.describe_model(),
            "device": self._device or self._resolve_device_name(),
            "threshold": self.config.threshold,
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

    # loading

    def warmup(self) -> None:
        """Load the model now; raises `Sam3Unavailable` on failure."""
        with self._lock:
            self._ensure_loaded()

    def _ensure_loaded(self) -> Tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._processor
        if self._error is not None:
            raise Sam3Unavailable(self._error)
        try:
            import torch

            device = self._resolve_device_name()
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise Sam3Unavailable(
                    f"SAM3_DEVICE={device!r} but no CUDA device is visible to torch."
                )
            self._model, self._processor = self._loader(self.config, device)
            self._device = device
            return self._model, self._processor
        except Sam3Unavailable as exc:
            self._error = str(exc)
            raise
        except Exception as exc:  # gated repo, OOM, corrupt download, ...
            message = f"{type(exc).__name__}: {exc}"
            lowered = message.lower()
            if any(k in lowered for k in ("gated", "401", "403", "authoriz", "token")):
                message += (
                    " — the SAM 3 weights are gated: request access to "
                    f"https://huggingface.co/{DEFAULT_HF_MODEL} and run `hf auth login`."
                )
            self._error = message
            raise Sam3Unavailable(message) from exc

    # inference

    @staticmethod
    def _decode_image(image_bytes: bytes) -> Any:
        import cv2
        from PIL import Image

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            raise PromptError("The uploaded frame could not be decoded as an image.")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def segment_with_points(
        self,
        image_bytes: bytes,
        points: Sequence[PromptPoint],
        *,
        text: Optional[str] = None,
        image_key: Optional[str] = None,
        refine: Optional[BoxRefiner] = None,
    ) -> ConceptResult:
        """Click-driven entry point used by the API.

        Converts `points` to exemplar boxes
        and runs `segment`. Either `points` or `text` must be given.
        """
        text = (text or "").strip() or None
        if not points and text is None:
            raise PromptError("Click on an example of the class or type its name.")
        image = self._decode_image(image_bytes)
        width, height = image.size
        boxes = exemplar_boxes_from_points(
            points, width, height, refine=refine, fraction=self.config.exemplar_fraction
        )
        if not boxes and text is None:
            raise PromptError("No usable exemplar could be built from the clicks.")
        return self.segment(
            image_bytes, text=text, boxes=boxes, image_key=image_key, image=image
        )

    def segment(
        self,
        image_bytes: bytes,
        *,
        text: Optional[str] = None,
        boxes: Sequence[ExemplarBox] = (),
        image_key: Optional[str] = None,
        image: Any = None,
    ) -> ConceptResult:
        """Run SAM 3 with a text prompt and/or exemplar boxes; union the hits.

        `image_key` identifies the frame for the embedding cache (defaults to
        the SHA-1 of the bytes). `image` may pass an already decoded PIL image.
        """
        text = (text or "").strip() or None
        if text is None and not boxes:
            raise PromptError("A text prompt or at least one exemplar box is required.")
        key = image_key or hashlib.sha1(image_bytes).hexdigest()

        with self._lock:
            model, processor = self._ensure_loaded()
            import torch

            reuse = key == self._last_image_key and self._last_vision_embeds is not None
            if reuse:
                image = self._last_image
            elif image is None:
                image = self._decode_image(image_bytes)
            width, height = image.size
            for box in boxes:
                box.validate(width, height)

            prompt_kwargs: Dict[str, Any] = {}
            if boxes:
                prompt_kwargs["input_boxes"] = [[b.as_list() for b in boxes]]
                prompt_kwargs["input_boxes_labels"] = [[b.label for b in boxes]]
            inputs = processor(
                images=image, text=text, return_tensors="pt", **prompt_kwargs
            )
            device = self._device or "cpu"
            model_dtype = next(model.parameters()).dtype
            tensors = {
                k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")
            }
            pixel_values = tensors.pop("pixel_values").to(model_dtype)
            original_sizes = inputs["original_sizes"]
            if hasattr(original_sizes, "tolist"):
                original_sizes = original_sizes.tolist()

            encoder_ms = 0.0
            with torch.inference_mode():
                if reuse:
                    vision_embeds = self._last_vision_embeds
                else:
                    started = time.perf_counter()
                    vision_embeds = model.get_vision_features(pixel_values=pixel_values)
                    encoder_ms = (time.perf_counter() - started) * 1000
                    self._last_image_key = key
                    self._last_image = image
                    self._last_vision_embeds = vision_embeds

                forward_kwargs = {
                    k: tensors[k]
                    for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels")
                    if k in tensors
                }
                if "input_boxes" in forward_kwargs:
                    forward_kwargs["input_boxes"] = forward_kwargs["input_boxes"].to(model_dtype)
                started = time.perf_counter()
                outputs = model(vision_embeds=vision_embeds, **forward_kwargs)
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.config.threshold,
                    mask_threshold=self.config.mask_threshold,
                    target_sizes=[tuple(int(v) for v in original_sizes[0])],
                )[0]
                decoder_ms = (time.perf_counter() - started) * 1000

        masks = results["masks"]
        scores = results["scores"]
        masks_np = (
            masks.detach().to("cpu").numpy() if hasattr(masks, "detach") else np.asarray(masks)
        )
        scores_np = (
            scores.detach().float().to("cpu").numpy()
            if hasattr(scores, "detach")
            else np.asarray(scores, dtype=np.float32)
        ).reshape(-1)
        if masks_np.ndim == 3 and masks_np.shape[0] > 0:
            union = masks_np.astype(bool).any(axis=0)
        else:
            union = np.zeros((height, width), dtype=bool)
        instance_scores = [float(s) for s in scores_np]
        return ConceptResult(
            mask=union,
            score=max(instance_scores) if instance_scores else 0.0,
            height=height,
            width=width,
            encoder_ms=encoder_ms,
            decoder_ms=decoder_ms,
            embedding_reused=reuse,
            instances=int(len(instance_scores)),
            instance_scores=instance_scores,
            exemplars=list(boxes),
        )

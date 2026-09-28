"""SAM 3 image segmentation: one frame in, a concept mask out.

Prompts are either a class name ("coral"), click points turned into exemplar
boxes, or both. The returned mask is the union of every detected region, so the
reviewer paints a whole class in one shot rather than one object at a time.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from src.core.config import DEFAULT_SAM3_MODEL, Settings
from src.core.errors import Unavailable
from src.domain.images import decode_image_pil
from src.domain.segmentation import (
    ConceptResult,
    ExemplarBox,
    PromptError,
    PromptPoint,
    exemplar_boxes_from_points,
)

ModelLoader = Callable[["Sam3Config", str], Tuple[Any, Any]]


class Sam3Unavailable(Unavailable):
    """SAM 3 cannot be used (not installed, gated weights, load failure)."""


@dataclass
class Sam3Config:
    """Everything the service needs to load and run the model."""

    hf_model: str = DEFAULT_SAM3_MODEL
    device: str = "auto"
    dtype: str = "auto"
    threshold: float = 0.5
    mask_threshold: float = 0.5
    exemplar_fraction: float = 0.06
    eager: bool = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "Sam3Config":
        return cls(
            hf_model=settings.sam3_model,
            device=settings.sam3_device,
            dtype=settings.sam3_dtype,
            threshold=settings.sam3_threshold,
            mask_threshold=settings.sam3_mask_threshold,
            exemplar_fraction=settings.sam3_exemplar_fraction,
            eager=settings.sam3_eager,
        )

    def describe_model(self) -> str:
        """Short model name for status responses."""
        return (
            os.path.basename(self.hf_model.rstrip("/"))
            if os.path.isdir(self.hf_model)
            else self.hf_model
        )


def _default_loader(config: Sam3Config, device: str) -> Tuple[Any, Any]:
    """Load `Sam3Model` + `Sam3Processor` (weights are cached by the Hub)."""
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
    """Lazy, lock-protected SAM 3 wrapper with a one-frame embedding cache.

    The cache is what makes click-by-click refinement cheap: re-prompting the
    same frame reuses the vision embeddings and only re-runs the decoder, which
    is reported back as `embedding_reused` plus per-stage timings.
    """

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
        self._last_image: Any = None  # PIL image, needed to normalise boxes
        self._last_vision_embeds: Any = None

    # status

    @staticmethod
    def installed() -> bool:
        try:
            import torch  # noqa: F401
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
                "(pip install -r backend/requirements.txt)."
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
        """Must be called with `self._lock` held."""
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
                    f"https://huggingface.co/{DEFAULT_SAM3_MODEL} and run `hf auth login`."
                )
            self._error = message
            raise Sam3Unavailable(message) from exc

    # inference

    def segment_with_points(
        self,
        image_bytes: bytes,
        points: Sequence[PromptPoint],
        *,
        text: Optional[str] = None,
        image_key: Optional[str] = None,
    ) -> ConceptResult:
        """Click-driven entry point used by the API.

        Each click becomes a small exemplar box; a text prompt can be combined
        with them. Either `points` or `text` must be given.
        """
        text = (text or "").strip() or None
        if not points and text is None:
            raise PromptError("Click on an example of the class or type its name.")
        image = decode_image_pil(image_bytes)
        width, height = image.size
        boxes = exemplar_boxes_from_points(
            points,
            width,
            height,
            fraction=self.config.exemplar_fraction,
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

        `image_key` identifies the frame for the embedding cache (defaults to the
        SHA-1 of the bytes). `image` may pass an already decoded PIL image.
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
                image = decode_image_pil(image_bytes)
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
            tensors = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
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
                    for k in (
                        "input_ids",
                        "attention_mask",
                        "input_boxes",
                        "input_boxes_labels",
                    )
                    if k in tensors
                }
                if "input_boxes" in forward_kwargs:
                    forward_kwargs["input_boxes"] = forward_kwargs["input_boxes"].to(
                        model_dtype
                    )
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
            masks.detach().to("cpu").numpy()
            if hasattr(masks, "detach")
            else np.asarray(masks)
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

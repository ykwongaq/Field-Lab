"""Lazy, single-owner access to the SAM 3 models.

Two different models are needed and they are not small:

* the **image** model (a `Sam3Image` detector plus, when instance interactivity is
  on, a tracker) for turning a point/box/text prompt into a mask;
* the **video** predictor (detector + tracker, multi-GPU aware) for propagating an
  anchor mask through a clip.

Keeping both resident is the comfortable thing to do, but it is roughly two 848M
parameter models plus activations, which does not fit on a modest GPU alongside
the frame cache. So this manager is an LRU of size `max_resident`: when a request
needs a model that is not loaded, the least recently used *other* one is released
first. Set `max_resident = 2` when there is headroom, or pin each model to its own
device.

Nothing here imports torch or sam3 at module import time: the package is an
optional, heavy dependency, and the API must still answer `/status` (and the rest
of the app must still start) on a machine where it is not installed.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from src.core.errors import Unavailable

logger = logging.getLogger("vsr")

#: Relative model paths resolve against the backend root, not the process working
#: directory: the service is started either from `backend/` or from the repo root,
#: and a checkpoint should not move because of that.
BACKEND_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_path(value: Optional[str]) -> Optional[str]:
    """Absolute path for a configured model file (relative paths use the backend root)."""
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = BACKEND_ROOT / candidate
    return str(candidate)


class Sam3Unavailable(Unavailable):
    """SAM 3 cannot be used (not installed, gated weights, load failure)."""


INSTALL_HINT = (
    "Install the official SAM 3 package and its weights first: "
    "`pip install -e <path to the sam3 checkout>` (see backend/requirements.txt), "
    "then `hf auth login` so the gated facebook/sam3 checkpoint can be downloaded."
)


class ModelKind(str, Enum):
    """Which of the two models a caller wants."""

    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True)
class Sam3Config:
    """How to load the models, and how many may be resident at once."""

    checkpoint: Optional[str] = None
    bpe_path: Optional[str] = None
    device: str = "auto"
    max_resident: int = 1
    enable_inst_interactivity: bool = True
    #: Run inference under bf16 autocast (see `autocast_enabled`).
    autocast: bool = True

    def resolved_device(self) -> str:
        """`device`, with `auto` resolved to cuda when torch can see one."""
        if self.device and self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # torch missing entirely
            return "cpu"

    def resolved_checkpoint(self) -> Optional[str]:
        """The checkpoint to load, or `None` to let the package fetch the gated one.

        A path that is configured but missing is an error worth naming precisely:
        left to the loader it surfaces as a bare `FileNotFoundError` from inside
        `torch.load`, which says nothing about which file or why it moved.
        """
        path = resolve_model_path(self.checkpoint)
        if path is None:
            return None
        if not Path(path).is_file():
            raise Sam3Unavailable(
                f"sam3.checkpoint is set to {self.checkpoint!r}, which resolved to "
                f"{path}, but no such file exists. Relative paths resolve against "
                f"{BACKEND_ROOT}; set an absolute path or clear the setting to use "
                "the `facebook/sam3` checkpoint from the Hugging Face cache."
            )
        return path

    def resolved_bpe_path(self) -> Optional[str]:
        """The tokenizer vocabulary, or `None` for the copy inside the package.

        The file ships with `sam3` (`sam3/assets/`), so it never *needs* to be
        configured. A configured path that does not exist is treated as unset
        rather than fatal: it is a leftover from an older config, and the packaged
        copy is exactly what that setting was trying to point at.
        """
        path = resolve_model_path(self.bpe_path)
        if path is None:
            return None
        if not Path(path).is_file():
            logger.warning(
                "sam3.bpe_path is set to %r (%s) but no such file exists; using "
                "the vocabulary bundled with the sam3 package instead.",
                self.bpe_path,
                path,
            )
            return None
        return path

    def autocast_enabled(self) -> bool:
        """Whether inference should run under bf16 autocast.

        The SAM 3 reference code wraps its own video entry points in
        `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`, and the tracker
        keeps a bf16 context of its own. Mixing the two conventions is what
        produces "mat1 and mat2 must have the same dtype, but got BFloat16 and
        Float": half a forward pass runs in bf16, so an activation comes out bf16
        and meets a weight that is still fp32. Putting a whole call — features,
        conditioning and the propagation stream together — under one context keeps
        them consistent.

        CPU is left alone: bf16 there is slow, and it buys nothing on a model that
        is only ever a fallback.
        """
        if not self.autocast:
            return False
        return self.resolved_device().startswith("cuda")

    def describe(self) -> str:
        """Short label for status responses."""
        if self.checkpoint:
            return Path(self.checkpoint.rstrip("/\\")).name
        return "facebook/sam3 (checkpoint from the Hugging Face cache)"


@dataclass
class _Resident:
    """One loaded model and when it was last handed out."""

    payload: Any
    loaded_at: float
    used_at: float
    device: str

    def touch(self) -> None:
        self.used_at = time.time()


class ModelManager:
    """Owns the SAM 3 models for the process, loading them on first use."""

    def __init__(self, config: Optional[Sam3Config] = None) -> None:
        self.config = config or Sam3Config()
        self._lock = threading.RLock()
        self._resident: Dict[ModelKind, _Resident] = {}
        self._error: Optional[str] = None

    # ── availability ────────────────────────────────────────────────────

    @staticmethod
    def installed() -> bool:
        """Whether the `sam3` package is importable (checked without importing it)."""
        try:
            return importlib.util.find_spec("sam3") is not None
        except (ImportError, ValueError):
            return False

    def status(self) -> Dict[str, Any]:
        """Report availability, which models are loaded, and where they run."""
        installed = self.installed()
        loaded = [kind.value for kind in self._resident]
        error = self._error
        if not installed and error is None:
            error = f"The `sam3` package is not installed. {INSTALL_HINT}"
        return {
            "available": installed and error is None,
            "loaded": bool(loaded),
            "loaded_models": loaded,
            "model": self.config.describe(),
            "checkpoint": self.config.resolved_checkpoint(),
            "device": self.config.resolved_device(),
            "error": error,
        }

    # ── loading ─────────────────────────────────────────────────────────

    def image(self) -> Tuple[Any, Any]:
        """The SAM 3 image model and its processor (loading it if needed)."""
        return self._acquire(ModelKind.IMAGE)

    def video(self) -> Any:
        """The SAM 3 video predictor (loading it if needed)."""
        return self._acquire(ModelKind.VIDEO)

    def warmup(self, kind: ModelKind = ModelKind.IMAGE) -> None:
        """Load a model now, so the first real request does not pay for it."""
        self._acquire(kind)

    @contextlib.contextmanager
    def inference_context(self) -> Iterator[None]:
        """One autocast context covering a whole model call.

        Balanced by construction, which also matters for the opposite hazard: an
        autocast context entered somewhere deep in the tracker and never exited
        stays active on the thread and silently changes the dtype of everything
        that follows. Entering our own context makes the dtype explicit instead of
        inherited.
        """
        if not self.config.autocast_enabled():
            yield
            return
        import torch

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield

    def release(self, kind: Optional[ModelKind] = None) -> None:
        """Drop one model (or all of them) and hand the memory back."""
        with self._lock:
            kinds = [kind] if kind is not None else list(self._resident)
            for entry_kind in kinds:
                resident = self._resident.pop(entry_kind, None)
                if resident is None:
                    continue
                payload = resident.payload
                if entry_kind is ModelKind.VIDEO:
                    self._shutdown_video(payload)
                elif isinstance(payload, tuple):
                    payload = None
            if kinds:
                _empty_cuda_cache()

    def shutdown(self) -> None:
        """Release everything (called on application shutdown)."""
        self.release()

    def _acquire(self, kind: ModelKind) -> Any:
        with self._lock:
            resident = self._resident.get(kind)
            if resident is not None:
                resident.touch()
                return resident.payload

            if self._error is not None:
                raise Sam3Unavailable(self._error)

            if not self.installed():
                self._error = f"The `sam3` package is not installed. {INSTALL_HINT}"
                raise Sam3Unavailable(self._error)

            self._make_room(kind)
            try:
                payload = self._load(kind)
            except Sam3Unavailable as exc:
                self._error = str(exc)
                raise
            except Exception as exc:  # gated repo, OOM, corrupt download, ...
                message = f"{type(exc).__name__}: {exc}"
                lowered = message.lower()
                if any(
                    key in lowered
                    for key in ("gated", "401", "403", "authoriz", "token", "login")
                ):
                    message += (
                        " — the SAM 3 weights are gated: request access at "
                        "https://huggingface.co/facebook/sam3 and run `hf auth login`."
                    )
                elif "out of memory" in lowered:
                    message += (
                        " — lower `sam3.max_resident_models` or move the image and "
                        "video models onto different devices."
                    )
                self._error = message
                raise Sam3Unavailable(message) from exc

            now = time.time()
            self._resident[kind] = _Resident(
                payload=payload,
                loaded_at=now,
                used_at=now,
                device=self.config.resolved_device(),
            )
            return payload

    def _make_room(self, kind: ModelKind) -> None:
        """Evict other models until the new one fits. Caller holds the lock."""
        limit = max(1, self.config.max_resident)
        while len(self._resident) >= limit:
            others = [k for k in self._resident if k is not kind]
            if not others:
                break
            victim = min(others, key=lambda k: self._resident[k].used_at)
            self.release(victim)

    def _load(self, kind: ModelKind) -> Any:
        device = self.config.resolved_device()
        if kind is ModelKind.VIDEO and device.startswith("cpu"):
            # The video predictor calls `.cuda()` unconditionally.
            raise Sam3Unavailable(
                "The SAM 3 video predictor needs a CUDA device; set "
                "``sam3.device`` to a GPU (or `cuda:0`)."
            )

        checkpoint = self.config.resolved_checkpoint()
        bpe = self.config.resolved_bpe_path()

        if kind is ModelKind.IMAGE:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            model = build_sam3_image_model(
                bpe_path=bpe,
                device=device,
                eval_mode=True,
                checkpoint_path=checkpoint,
                load_from_HF=checkpoint is None,
                enable_segmentation=True,
                enable_inst_interactivity=self.config.enable_inst_interactivity,
            )
            processor = Sam3Processor(model, device=device)
            return (model, processor)

        from sam3.model_builder import build_sam3_video_predictor

        return build_sam3_video_predictor(
            checkpoint_path=checkpoint,
            bpe_path=bpe,
        )

    @staticmethod
    def _shutdown_video(predictor: Any) -> None:
        """Release a video predictor's worker processes and CUDA context."""
        shutdown = getattr(predictor, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:  # a half-initialised predictor must not break eviction
                pass


def _empty_cuda_cache() -> None:
    """Return freed CUDA blocks to the driver, when torch is around."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

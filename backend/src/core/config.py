"""Application settings.

Every tunable value is read from the environment exactly once, here. Nothing
else in the codebase should call `os.environ`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024**3  # 4 GiB source video
DEFAULT_MAX_FRAME_BYTES = 32 * 1024**2  # 32 MiB single frame
DEFAULT_SAM3_MODEL = "facebook/sam3"
DEFAULT_SAM3_THRESHOLD = 0.5
DEFAULT_SAM3_MASK_THRESHOLD = 0.5
DEFAULT_SAM3_EXEMPLAR_FRACTION = 0.06
DEFAULT_PROPAGATE_MAX_FRAMES = 120
DEFAULT_LOG_LEVEL = "INFO"

_TRUTHY = {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUTHY


def _list_env(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or default


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    # HTTP limits
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
    cors_origins: Tuple[str, ...] = ("*",)
    log_level: str = DEFAULT_LOG_LEVEL

    # SAM 3 (image concept segmentation)
    sam3_model: str = DEFAULT_SAM3_MODEL
    sam3_device: str = "auto"
    sam3_dtype: str = "auto"
    sam3_threshold: float = DEFAULT_SAM3_THRESHOLD
    sam3_mask_threshold: float = DEFAULT_SAM3_MASK_THRESHOLD
    sam3_exemplar_fraction: float = DEFAULT_SAM3_EXEMPLAR_FRACTION
    sam3_eager: bool = False

    # SAM 3 video tracker (mask propagation)
    sam3_tracker_model: Optional[str] = None
    propagate_max_frames: int = DEFAULT_PROPAGATE_MAX_FRAMES


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build `Settings` from the environment (cached; call with care in tests)."""
    return Settings(
        max_upload_bytes=_int_env("VSR_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
        max_frame_bytes=_int_env("VSR_MAX_FRAME_BYTES", DEFAULT_MAX_FRAME_BYTES),
        cors_origins=_list_env("VSR_CORS_ORIGINS", ("*",)),
        log_level=os.environ.get("VSR_LOG_LEVEL", DEFAULT_LOG_LEVEL),
        sam3_model=os.environ.get("SAM3_MODEL", DEFAULT_SAM3_MODEL),
        sam3_device=os.environ.get("SAM3_DEVICE", "auto"),
        sam3_dtype=os.environ.get("SAM3_DTYPE", "auto"),
        sam3_threshold=_float_env("SAM3_THRESHOLD", DEFAULT_SAM3_THRESHOLD),
        sam3_mask_threshold=_float_env(
            "SAM3_MASK_THRESHOLD", DEFAULT_SAM3_MASK_THRESHOLD
        ),
        sam3_exemplar_fraction=_float_env(
            "SAM3_EXEMPLAR_BOX", DEFAULT_SAM3_EXEMPLAR_FRACTION
        ),
        sam3_eager=_bool_env("SAM3_EAGER"),
        sam3_tracker_model=os.environ.get("SAM3_TRACKER_MODEL") or None,
        propagate_max_frames=_int_env(
            "PROPAGATE_MAX_FRAMES", DEFAULT_PROPAGATE_MAX_FRAMES
        ),
    )

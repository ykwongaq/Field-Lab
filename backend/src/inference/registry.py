"""Process-wide access to the heavy inference services.

The services are built lazily and cached, so the models are instanced once per
process (not once per request, and not at import time). The API layer depends on
these accessors rather than on module-level globals, which keeps the routers
testable: override the dependency instead of monkey-patching a module.

One `ModelManager` is shared by the image service and the propagator, which is
what makes the LRU eviction meaningful: the two models compete for the same
budget, and whichever was used least recently is the one to give up.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from src.core.config import get_settings
from src.core.jobs import JobRegistry
from src.inference.models import ModelKind, ModelManager, Sam3Config
from src.inference.sam3_image import Sam3ImageService
from src.inference.sam3_video import (
    PropagationRunner,
    PropagateConfig,
    Sam3VideoPropagator,
)

_manager: Optional[ModelManager] = None


def _model_config() -> Sam3Config:
    settings = get_settings()
    return Sam3Config(
        checkpoint=settings.sam3_checkpoint,
        bpe_path=settings.sam3_bpe_path,
        device=settings.sam3_device,
        max_resident=settings.sam3_max_resident_models,
        enable_inst_interactivity=settings.sam3_inst_interactivity,
        autocast=settings.sam3_autocast,
    )


def model_manager() -> ModelManager:
    """The single model budget every service draws from (created on first use)."""
    global _manager
    if _manager is None:
        _manager = ModelManager(_model_config())
    return _manager


@lru_cache(maxsize=1)
def get_sam3_service() -> Sam3ImageService:
    """The SAM 3 image segmentation service (lazily loads its weights)."""
    settings = get_settings()
    return Sam3ImageService(
        _model_config(),
        manager=model_manager(),
        cache_size=settings.sam3_image_cache_size,
        cache_clients=settings.sam3_image_cache_clients,
    )


@lru_cache(maxsize=1)
def get_propagator() -> Sam3VideoPropagator:
    """The windowed mask propagator (lazily loads the video predictor)."""
    return Sam3VideoPropagator(
        PropagateConfig.from_settings(get_settings()),
        manager=model_manager(),
        model_config=_model_config(),
    )


@lru_cache(maxsize=1)
def get_job_registry() -> JobRegistry:
    """The propagation job queue, with `PropagationRunner` as its worker."""
    settings = get_settings()
    return JobRegistry(
        PropagationRunner(get_propagator(), settings),
        max_jobs=settings.propagate_max_jobs,
        max_jobs_per_client=settings.propagate_max_jobs_per_client,
        ttl_seconds=settings.propagate_job_ttl_seconds,
    )


def warmup_models() -> list:
    """Load the models at start-up and keep them resident.

    Both are loaded when the budget allows it, which is the point of an eager
    start: the first click and the first propagation then cost nothing extra, and
    neither load can evict the other. With a budget of 1 the image model is the
    one to hold — prompts are far more frequent than propagation runs — and the
    tracker loads (evicting it) the first time a mask is propagated.
    """
    manager = model_manager()
    budget = get_settings().sam3_max_resident_models
    kinds = [ModelKind.IMAGE]
    if budget > 1:
        kinds.append(ModelKind.VIDEO)
    loaded = []
    for kind in kinds:
        manager.warmup(kind)
        loaded.append(kind.value)
    return loaded


def reset_services() -> None:
    """Drop the cached services (used by tests)."""
    global _manager
    get_sam3_service.cache_clear()
    get_propagator.cache_clear()
    get_job_registry.cache_clear()
    _manager = None


def shutdown_services() -> None:
    """Cancel queued work and release every model (used on shutdown).

    Only touches services that were actually built: creating a job registry at
    shutdown time would start a worker thread purely to stop it again.
    """
    try:
        if get_job_registry.cache_info().currsize:
            get_job_registry().shutdown()
    finally:
        manager = _manager
        if manager is not None:
            manager.shutdown()
        reset_services()

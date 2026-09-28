"""Process-wide access to the heavy inference services.

The services are built lazily and cached, so the models are instanced once per
process (not once per request, and not at import time). The API layer depends on
these accessors rather than on module-level globals, which keeps the routers
testable: override the dependency instead of monkey-patching a module.
"""

from __future__ import annotations

from functools import lru_cache

from src.core.config import get_settings
from src.inference.propagate import PropagateConfig, PropagateService
from src.inference.sam3 import Sam3Config, Sam3Service


@lru_cache(maxsize=1)
def get_sam3_service() -> Sam3Service:
    """The SAM 3 image segmentation service (lazily loads its weights)."""
    return Sam3Service(Sam3Config.from_settings(get_settings()))


@lru_cache(maxsize=1)
def get_propagate_service() -> PropagateService:
    """The mask propagation service (lazily loads the video tracker)."""
    settings = get_settings()
    return PropagateService(
        PropagateConfig.from_settings(settings),
        tracker_config=Sam3Config.from_settings(settings),
    )


def reset_services() -> None:
    """Drop the cached services (used by tests)."""
    get_sam3_service.cache_clear()
    get_propagate_service.cache_clear()

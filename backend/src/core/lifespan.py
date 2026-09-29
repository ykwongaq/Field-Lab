"""Application startup and shutdown."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.core.config import get_settings
from src.core.errors import Unavailable
from src.core.storage import ensure_dir
from src.inference.registry import get_sam3_service

logger = logging.getLogger("vsr")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Create the configured directories, optionally preload SAM 3, then serve.

    A failed preload is only logged: the API stays up so `/api/sam3/status` can
    explain what is wrong instead of the whole process refusing to start.
    """
    settings = get_settings()
    for directory in (settings.log_dir, settings.temp_dir, settings.projects_dir):
        ensure_dir(directory)
    if settings.enable_sam3 and settings.sam3_eager:
        try:
            get_sam3_service().warmup()
        except Unavailable as exc:
            logger.warning("SAM 3 could not be preloaded: %s", exc)
    yield

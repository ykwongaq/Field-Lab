"""Application startup and shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI

from src.core.config import Settings, get_settings
from src.core.errors import Unavailable
from src.core.sessions import sweep_sessions
from src.core.storage import ensure_dir
from src.domain.extract import missing_binaries
from src.inference.registry import (
    warmup_models,
    shutdown_services,
)

logger = logging.getLogger("vsr")

#: How often idle sessions are reaped while the service is serving.
SWEEP_INTERVAL_SECONDS = 15 * 60


def sweep(settings: Settings) -> List[str]:
    """Drop idle and over-quota sessions, returning the ids removed."""
    return sweep_sessions(
        settings.temp_dir,
        settings.session_ttl_seconds,
        max_bytes=settings.session_max_bytes,
    )


async def _sweep_forever(settings: Settings) -> None:
    """Keep reaping sessions for as long as the app is serving.

    The page-unload beacon covers the ordinary case; this covers the ones it
    cannot — a crashed browser, a killed tab, a closed laptop — so `temp_dir`
    cannot quietly fill up. A failed sweep is logged and retried, never fatal.
    """
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        try:
            dropped = await asyncio.to_thread(sweep, settings)
        except Exception:  # noqa: BLE001 - a sweep must not take the app down
            logger.exception("Session sweep failed")
            continue
        if dropped:
            logger.info("Swept %d idle session(s)", len(dropped))


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Prepare the directories, preload SAM 3, reap leftovers, then serve.

    A failed preload is only logged: the API stays up so `/api/sam3/status` can
    explain what is wrong instead of the whole process refusing to start.
    """
    settings = get_settings()
    for directory in (settings.log_dir, settings.temp_dir, settings.projects_dir):
        ensure_dir(directory)

    # A crashed run cannot clean up after itself, so the first sweep is on the
    # way up. Sessions used within the TTL survive a restart untouched, which
    # matters because a restart does not invalidate the frames on disk.
    dropped = await asyncio.to_thread(sweep, settings)
    if dropped:
        logger.info("Removed %d leftover session(s)", len(dropped))

    # A project whose frames must be decoded needs these, so say so now rather
    # than when someone uploads one.
    missing = missing_binaries(settings.ffmpeg_bin, settings.ffprobe_bin)
    if missing:
        logger.warning(
            "%s not found on PATH, so a project that needs its frames extracted "
            "cannot be opened. Install ffmpeg, or set VSR_FFMPEG_BIN / "
            "VSR_FFPROBE_BIN to the binary's absolute path and restart the "
            "service.",
            " and ".join(missing),
        )

    if settings.enable_sam3 and settings.sam3_eager:
        try:
            loaded = warmup_models()
            logger.info(
                "SAM 3 preloaded and held resident: %s", ", ".join(loaded) or "nothing"
            )
        except Unavailable as exc:
            logger.warning("SAM 3 could not be preloaded: %s", exc)
        except Exception as exc:  # noqa: BLE001 - never refuse to start for this
            logger.warning("SAM 3 preload failed: %s: %s", type(exc).__name__, exc)

    sweeper = asyncio.create_task(_sweep_forever(settings))
    try:
        yield
    finally:
        sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper
        # Cancel queued propagations and hand the models back before the process
        # goes away: a reload should not leave a GPU holding two checkpoints.
        await asyncio.to_thread(shutdown_services)

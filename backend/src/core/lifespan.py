"""Application startup and shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI
from src.core.config import Settings, get_settings
from src.core.sessions import sweep_sessions
from src.core.singleton import LOCK_NAME
from src.core.singleton import acquire as acquire_instance_lock
from src.core.storage import ensure_dir
from src.domain.extract import missing_binaries
from src.inference.registry import shutdown_services, warmup_models

logger = logging.getLogger("vsr")

#: How often idle sessions are reaped while the service is serving.
SWEEP_INTERVAL_SECONDS = 15 * 60


def sweep(settings: Settings) -> List[str]:
    """Drop idle and over-quota sessions, returning the ids removed.

    The per-owner cap runs before the global one, so an owner over its own budget
    loses its own sessions and never a colleague's (see `sweep_sessions`).
    """
    return sweep_sessions(
        settings.temp_dir,
        settings.session_ttl_seconds,
        max_bytes=settings.session_max_bytes,
        max_bytes_per_owner=settings.session_max_bytes_per_client,
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

    SAM 3 is loaded here, before the service accepts a single request, so the
    first prompt and the first propagation find the models already resident. A
    preload failure is fatal on purpose: the process refuses to start rather
    than come up without the model it exists to serve.

    A *second process*, on the other hand, is refused outright. The model gate,
    the job queue and the session sweeper are all per process, so a second one
    sharing this store would break the first rather than add capacity; see
    `core.singleton`.
    """
    settings = get_settings()
    for directory in (settings.log_dir, settings.temp_dir, settings.projects_dir):
        ensure_dir(directory)

    # Taken before anything else touches the store, so nothing can be swept or
    # queued twice. Runs in a thread because the handover grace period sleeps.
    instance = await asyncio.to_thread(
        acquire_instance_lock, os.path.join(settings.temp_dir, LOCK_NAME)
    )
    try:
        async with _serving(settings):
            yield
    finally:
        instance.release()


@asynccontextmanager
async def _serving(settings: Settings) -> AsyncIterator[None]:
    """Bring the service up and hold it up, with the instance lock already held."""
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

    # Loaded unconditionally, on the way up: there is no lazy fallback and no
    # "enabled"/"eager" gate. If the models cannot be loaded the process must not
    # start, so a failure here propagates and stops the service.
    loaded = warmup_models()
    logger.info("SAM 3 preloaded and held resident: %s", ", ".join(loaded) or "nothing")

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

"""Turns domain errors into HTTP responses.

Routers raise plain exceptions (`PromptError`, `PropagateError`,
`PayloadTooLarge`, ...) and each class declares its own status code, so no route
needs a `try/except` ladder mapping exceptions to `HTTPException`.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.errors import Unavailable, VsrError

logger = logging.getLogger("vsr")


def register_exception_handlers(app: FastAPI) -> None:
    """Install one handler for the whole `VsrError` hierarchy."""

    @app.exception_handler(VsrError)
    async def _handle_vsr_error(_request: Request, exc: VsrError) -> JSONResponse:
        if isinstance(exc, Unavailable):
            # 503 covers deliberate states (a disabled feature, an unloaded
            # model), so it is logged as a warning rather than an error.
            logger.warning("Unavailable: %s", exc.message)
        elif exc.status_code >= 500:
            logger.error("Backend error: %s", exc.message)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

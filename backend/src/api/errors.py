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
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.message}
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Log an unhandled failure with its traceback, and answer 500.

        Without this the traceback reaches only uvicorn's stderr — not the app's
        own log, which is where an operator looks when `log_dir` is configured.
        A 500 from inside a model call is exactly the case where the traceback is
        the only clue, and it is worth having it next to the requests that led to
        it. The response body stays generic on purpose: a stack trace in a
        response is an information leak, so the detail lives in the log.
        """
        logger.error(
            "Unhandled %s while serving %s %s",
            type(exc).__name__,
            request.method,
            request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500, content={"detail": "Internal Server Error"}
        )

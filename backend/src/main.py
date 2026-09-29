"""FastAPI application for the VideoSegmenter backend.

Run it from `backend/`:

    uvicorn src.main:app --reload --port 8000

This module only wires things together — settings, middleware, exception
handlers and routers. All behaviour lives in `api/`, `domain/`, `projects/` and
`inference/`.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.errors import register_exception_handlers
from src.api.routes import health, masks, propagate, sam3, sessions
from src.core.config import get_settings
from src.core.lifespan import lifespan


def create_app() -> FastAPI:
    """Build the application: middleware, exception handlers, routers."""
    settings = get_settings()
    logging.getLogger("vsr").setLevel(settings.log_level.upper())

    app = FastAPI(title="Video Segmenter backend", lifespan=lifespan)

    # The frontend talks to this service from the Vite dev origin (or a hosted
    # origin), so allow cross-origin requests. No credentials are used.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(masks.router)
    app.include_router(sam3.router)
    app.include_router(propagate.router)
    app.include_router(sessions.router)
    return app


app = create_app()

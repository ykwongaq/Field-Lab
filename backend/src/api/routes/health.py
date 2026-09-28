"""Liveness probe."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Cheap check that the service is up (touches no model)."""
    return {"status": "ok"}

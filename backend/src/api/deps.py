"""Shared FastAPI plumbing: upload handling and dependencies."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Depends, Header, UploadFile

from src.core.config import Settings, get_settings
from src.core.errors import InvalidRequest, PayloadTooLarge, Unavailable
from src.core.identity import CLIENT_HEADER, validate_client_id
from src.core.jobs import JobRegistry
from src.inference.registry import get_job_registry, get_propagator, get_sam3_service
from src.inference.sam3_image import Sam3ImageService
from src.inference.sam3_video import Sam3VideoPropagator

CHUNK_BYTES = 8 * 1024 * 1024


async def stream_upload_to_path(
    upload: UploadFile,
    path: str,
    limit: int,
    *,
    label: str = "Upload",
    chunk_bytes: int = CHUNK_BYTES,
) -> int:
    """Stream an upload to disk, refusing anything above `limit` bytes.

    Source videos can be gigabytes, so they are written in chunks and never held
    in memory. Returns the number of bytes written.
    """
    written = 0
    with open(path, "wb") as sink:
        while True:
            data = await upload.read(chunk_bytes)
            if not data:
                break
            written += len(data)
            if limit and written > limit:
                raise PayloadTooLarge(
                    f"{label} exceeds the upload limit of {limit} bytes"
                )
            sink.write(data)
    await upload.close()
    if written == 0:
        raise InvalidRequest(f"{label} is empty")
    return written


SAM3_DISABLED_MESSAGE = (
    "SAM 3 is disabled by configuration (`sam3.enabled` is false in "
    "config/server.json; set SAM3_ENABLED=1 to turn it back on)."
)


def require_sam3(settings: Settings) -> None:
    """Refuse a SAM 3 request early when the model is switched off.

    Called before any session is opened or any service is touched, so a disabled
    backend never imports torch, let alone loads weights.
    """
    if not settings.enable_sam3:
        raise Unavailable(SAM3_DISABLED_MESSAGE)


def client_id(
    x_vsr_client: Annotated[Optional[str], Header(alias=CLIENT_HEADER)] = None,
) -> str:
    """The caller's partition key, from the `X-Vsr-Client` header.

    Required on every endpoint that touches a session, a prompt or a job, because
    that is the only thing that says whose frames, masks and runs a request is
    allowed to see. See `core.identity` for what it does and, more importantly,
    what it does not do.
    """
    return validate_client_id(x_vsr_client)


SettingsDep = Annotated[Settings, Depends(get_settings)]
ClientDep = Annotated[str, Depends(client_id)]
Sam3Dep = Annotated[Sam3ImageService, Depends(get_sam3_service)]
PropagateDep = Annotated[Sam3VideoPropagator, Depends(get_propagator)]
JobsDep = Annotated[JobRegistry, Depends(get_job_registry)]

"""Shared FastAPI plumbing: upload handling, JSON form fields, dependencies."""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import Depends, UploadFile

from src.core.config import Settings, get_settings
from src.core.errors import InvalidRequest, PayloadTooLarge
from src.inference.propagate import PropagateService
from src.inference.registry import get_propagate_service, get_sam3_service
from src.inference.sam3 import Sam3Service

CHUNK_BYTES = 8 * 1024 * 1024


async def read_capped_upload(
    upload: UploadFile, limit: int, *, label: str = "Upload"
) -> bytes:
    """Read an upload into memory, refusing anything above `limit` bytes.

    Only for single frames, which are capped at a few MiB. Source videos must go
    through `stream_upload_to_path` instead.
    """
    data = await upload.read(limit + 1)
    await upload.close()
    if not data:
        raise InvalidRequest(f"{label} is empty")
    if len(data) > limit:
        raise PayloadTooLarge(f"{label} exceeds the upload limit of {limit} bytes")
    return data


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


def parse_json_field(raw: str, field: str) -> Any:
    """Decode a JSON form field, reporting a 422 instead of a 500."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidRequest(f"`{field}` is not JSON: {exc}") from exc


SettingsDep = Annotated[Settings, Depends(get_settings)]
Sam3Dep = Annotated[Sam3Service, Depends(get_sam3_service)]
PropagateDep = Annotated[PropagateService, Depends(get_propagate_service)]

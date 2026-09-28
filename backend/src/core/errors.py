"""Error hierarchy shared by every layer.

Each class carries the HTTP status the API should answer with, so callers can
simply raise and let a single exception handler in the API layer turn it into a
response. This module deliberately imports nothing from FastAPI, so the project
builder and the CLIs can use these errors too.
"""

from __future__ import annotations


class VsrError(Exception):
    """Base class for every error this backend raises on purpose."""

    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequest(VsrError, ValueError):
    """The caller sent something we cannot act on (bad prompt, mask, video...).

    Also a `ValueError` so existing `except ValueError` code keeps working.
    """

    status_code = 422


class Unavailable(VsrError):
    """A required model or resource is not available right now."""

    status_code = 503


class PayloadTooLarge(VsrError):
    """An upload (or a batch of uploads) exceeded its configured limit."""

    status_code = 413


class UnsupportedMediaType(VsrError):
    """The upload is a kind of file this backend does not accept."""

    status_code = 415

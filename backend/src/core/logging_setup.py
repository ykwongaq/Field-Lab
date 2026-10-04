"""Logging setup: the `vsr` logger, optionally mirrored to `log_dir`.

Every module logs through the `vsr` logger, so configuring it here is the only
place that decides where a record goes. Records always reach stderr, which is
what uvicorn already captures; when `log_dir` is set they are written to a
rotating file as well, so an unattended propagation run leaves something behind
to read afterwards.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from src.core.config import Settings
from src.core.storage import ensure_dir

LOGGER_NAME = "vsr"
LOG_FILENAME = "backend.log"

#: Rotate at 5 MiB and keep three files, so logging cannot grow without bound.
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

#: Marks the handler this module owns. The app can be built more than once in a
#: process (tests do), so re-configuring must not attach a second file handler.
_FILE_HANDLER_NAME = "vsr-file"


def configure_logging(settings: Settings) -> Optional[str]:
    """Set the `vsr` logger's level and, when configured, add a rotating file.

    Returns the log file's path, or `None` when logging only goes to stderr.
    Safe to call more than once: the file handler is attached at most once.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(settings.log_level.upper())
    if not settings.log_dir:
        return None

    path = os.path.join(ensure_dir(settings.log_dir), LOG_FILENAME)
    if any(handler.name == _FILE_HANDLER_NAME for handler in logger.handlers):
        return path

    handler = RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    handler.name = _FILE_HANDLER_NAME
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    return path

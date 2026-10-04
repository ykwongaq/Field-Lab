"""Filesystem layout: the configured directories and disposable scratch space.

`temp_dir` and `projects_dir` come from `core.config`; this module is the only
place that creates or removes them, so "clean up after yourself" is one policy
instead of a rule every caller has to remember.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from typing import Iterator


def ensure_dir(path: str) -> str:
    """Create `path` (and its parents) when needed and return it.

    An empty path means "not configured" and is left alone, so a blank
    `log_dir`/`temp_dir` disables that directory instead of creating the CWD.
    """
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def remove_tree(path: str) -> None:
    """Delete a directory tree, tolerating a path that is already gone."""
    if path:
        shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def scratch_dir(root: str, prefix: str = "vsr-") -> Iterator[str]:
    """Yield a fresh directory inside `root` and always delete it again.

    Cleanup happens on the way out — including when the body raises — so a
    failed request cannot leave uploads behind in the temp directory.
    """
    ensure_dir(root)
    path = tempfile.mkdtemp(prefix=prefix, dir=root or None)
    try:
        yield path
    finally:
        remove_tree(path)

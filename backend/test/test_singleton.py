"""Checks for the single-instance lock that keeps the backend to one process.

The lock is what stops `uvicorn --workers N` from quietly giving every worker its
own model gate, job queue and session sweeper. Nothing here needs a GPU or a
model: the guarantee is an OS file lock, and that is what is exercised.
"""

from __future__ import annotations

import os

import pytest

from src.core.singleton import LOCK_NAME, AlreadyRunning, acquire


def lock_path(scratch, name: str = LOCK_NAME) -> str:
    return os.path.join(str(scratch), name)


def test_a_second_holder_is_refused(scratch):
    """The whole point: a second process fails rather than starting anyway."""
    path = lock_path(scratch)
    first = acquire(path, grace_seconds=0)
    try:
        with pytest.raises(AlreadyRunning, match="single process"):
            acquire(path, grace_seconds=0)
    finally:
        first.release()


def test_releasing_lets_the_next_one_in(scratch):
    """A process that exited cleanly must not lock the store out for ever."""
    path = lock_path(scratch)
    acquire(path, grace_seconds=0).release()

    second = acquire(path, grace_seconds=0)
    try:
        assert second.path == path
    finally:
        second.release()


def test_release_is_idempotent(scratch):
    """Shutdown paths run twice on some exits, and must not raise."""
    path = lock_path(scratch)
    lock = acquire(path, grace_seconds=0)
    lock.release()
    lock.release()

    after = acquire(path, grace_seconds=0)
    after.release()


def test_the_holder_is_recorded(scratch):
    """The pid is written so an operator can tell who owns the store.

    Read back after the release, because Windows locks bytes *mandatorily*: while
    the lock is held, even a read of that byte is refused. That is precisely why
    the pid line is a convenience and the lock is the actual guarantee.
    """
    path = lock_path(scratch)
    acquire(path, grace_seconds=0).release()

    with open(path, encoding="ascii") as handle:
        assert int(handle.read().strip()) == os.getpid()


def test_a_stale_lock_file_does_not_block_startup(scratch):
    """The lock is an OS lock, not a marker file, so a crash cannot wedge it.

    Leftover files are exactly what a well-behaved run leaves behind — `release`
    closes the descriptor and the lock goes with it — so acquiring over an
    existing, unlocked file has to succeed.
    """
    path = lock_path(scratch)
    with open(path, "w", encoding="ascii") as handle:
        handle.write("999999\n")

    lock = acquire(path, grace_seconds=0)
    lock.release()


def test_separate_stores_do_not_conflict(scratch):
    """Only the same store is refused; the lock is scoped to what it protects."""
    one = acquire(lock_path(scratch, os.path.join("a", LOCK_NAME)), grace_seconds=0)
    two = acquire(lock_path(scratch, os.path.join("b", LOCK_NAME)), grace_seconds=0)
    try:
        assert one.path != two.path
    finally:
        one.release()
        two.release()

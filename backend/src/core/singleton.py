"""One process, or none.

Everything that makes this service safe under concurrency is *per process*:

* `ModelGate` is the only thing standing between two requests and one GPU, and a
  gate cannot see across processes;
* `JobRegistry` holds the queue and the cancellation flags in memory, so a job
  accepted by one worker is unknown to the next — a poll routed elsewhere is a
  404 and a cancel is a no-op;
* session ownership is enforced against a store the sweeper deletes from, so a
  second process's sweep can evict sessions the first is still serving;
* the two SAM 3 checkpoints are the same GPU memory, which two processes cannot
  both hold.

So `uvicorn --workers N` does not scale this service, it breaks it quietly: N
gates, N queues, and a reviewer whose propagation was accepted is told by the
next worker that the job does not exist. The GPU is the bottleneck and it cannot
be replicated, so a second process on the same store is refused at start-up
rather than discovered later as a lost job.

`--reload` is unaffected: uvicorn runs one supervisor plus a single worker, so
only one process ever holds the lock. `GRACE_SECONDS` exists for the handover
between reloads, where the replacement can come up before the old worker has
finished exiting.
"""

from __future__ import annotations

import os
import time

#: The lock file, inside the store this process owns.
LOCK_NAME = "backend.lock"

#: How long to keep retrying before declaring another instance is running. Long
#: enough for a reload handover, short enough that a real conflict is reported
#: promptly.
GRACE_SECONDS = 10.0
POLL_SECONDS = 0.1


class AlreadyRunning(RuntimeError):
    """Another VideoSegmenter process already owns this store."""


def _try_lock(fd: int) -> bool:
    """Take an exclusive, non-blocking OS lock on `fd`, or report that we cannot."""
    try:
        if os.name == "nt":
            import msvcrt

            # Locks one byte from the current position, which is 0 here.
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


class InstanceLock:
    """A held lock. Released explicitly, or by the OS when the process dies."""

    def __init__(self, fd: int, path: str) -> None:
        self._fd = fd
        self.path = path

    def release(self) -> None:
        """Drop the lock. Idempotent, and safe to call on a process that is exiting."""
        fd, self._fd = self._fd, -1
        if fd < 0:
            return
        try:
            if os.name == "nt":
                import msvcrt

                # Windows unlocks the byte range from the current position, so
                # the seek has to match the one `acquire` locked with.
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        finally:
            os.close(fd)


def acquire(
    path: str,
    *,
    grace_seconds: float = GRACE_SECONDS,
    poll_seconds: float = POLL_SECONDS,
) -> InstanceLock:
    """Take the single-instance lock at `path`, waiting briefly for a handover.

    Raises `AlreadyRunning` when another process still holds it.

    This is an OS file lock rather than a marker file on purpose: the kernel
    releases it when the holding process exits, however it exits, so a crash
    cannot leave a stale lock that stops the service from ever starting again.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while not _try_lock(fd):
        if time.monotonic() >= deadline:
            os.close(fd)
            raise AlreadyRunning(
                f"Another VideoSegmenter process already owns {path}. Run the "
                "backend as a single process (`uvicorn src.main:app`, with no "
                "`--workers`): the model gate, the propagation queue and the "
                "session store are all per process, and two processes would also "
                "contend for one GPU."
            )
        time.sleep(poll_seconds)

    # Record the holder so an operator looking at the directory can tell. The
    # lock is what enforces anything; this line is a convenience, so a failure to
    # write it is not worth refusing to start over.
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
    except OSError:
        pass

    return InstanceLock(fd, path)

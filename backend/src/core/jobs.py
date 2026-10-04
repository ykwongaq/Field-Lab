"""Background propagation jobs.

Propagating a mask over a whole clip takes minutes, so it cannot run inside the
request that asks for it: the reviewer needs to watch the masks arrive, cancel a
run that is going wrong, and queue a second object without blocking the first.

Two rules shape this module:

* **One job at a time.** A single worker thread consumes the queue, so a GPU
  never sees two propagations at once. That is what makes "one object per job"
  a memory guarantee rather than a hope.
* **Results are readable while running.** `PropagationJob.snapshot()` hands out a
  consistent copy of the masks produced so far, so a poller can paint frames as
  they land instead of waiting for the run to finish.

The registry is deliberately ignorant of SAM 3: it takes a `runner` callable, so
the whole job lifecycle is testable with a fake runner and no GPU.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from src.core.config import DEFAULT_PROPAGATE_MAX_JOBS_PER_CLIENT
from src.core.errors import InvalidRequest, NotFound


class JobState(str, Enum):
    """Where a job is in its life."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def finished(self) -> bool:
        return self in (JobState.DONE, JobState.CANCELLED, JobState.FAILED)


class JobCancelled(RuntimeError):
    """Raised inside a runner when the caller asked the job to stop."""


@dataclass
class JobProgress:
    """How far along a run is, in frames and in windows."""

    frames_done: int = 0
    frames_total: int = 0
    window_index: int = 0
    windows_total: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "window_index": self.window_index,
            "windows_total": self.windows_total,
        }


@dataclass
class PropagationJob:
    """One propagation run: what was asked for, and what it has produced."""

    id: str
    session_id: str
    #: The client that asked for the run. Every read and cancel has to present the
    #: same id, so one reviewer's jobs are invisible to another's. See
    #: `core.identity`.
    client_id: str
    anchor: int
    direction: str
    first: int
    last: int
    object_id: Optional[int] = None
    #: The mask to propagate, and how many frames the clip has. Both are needed by
    #: the runner but are not part of any response (the mask is large, and the
    #: frame count is only used to bound the plan).
    anchor_mask: Optional[np.ndarray] = None
    frame_count: int = 0
    #: Frames a human verified by hand, other than the anchor. They are seeded
    #: into the tracker as authoritative exactly like the anchor and are never
    #: overwritten by the run. Like `anchor_mask`, they stay out of every response.
    pins: Dict[int, np.ndarray] = field(default_factory=dict)
    #: `derived` | `verified`; `None` uses the configured default. See
    #: `domain.windows` for what a window with no verified frame may be seeded on.
    chaining: Optional[str] = None
    state: JobState = JobState.QUEUED
    error: Optional[str] = None
    progress: JobProgress = field(default_factory=JobProgress)
    plan: Optional[Dict[str, Any]] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    updated_at: float = 0.0

    #: Cooperative cancellation: runners poll this between frames and windows.
    cancel: threading.Event = field(default_factory=threading.Event)
    #: Guards `_masks` and `progress` (the worker writes, requests read).
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _masks: Dict[int, np.ndarray] = field(default_factory=dict)

    # ── runner-facing API (called from the worker thread) ───────────────

    def check_cancelled(self) -> None:
        """Raise when the caller has asked this job to stop."""
        if self.cancel.is_set():
            raise JobCancelled(f"Propagation job {self.id} was cancelled.")

    def set_plan(self, plan: Dict[str, Any]) -> None:
        """Record the window plan so callers can show the shape of the run."""
        with self._lock:
            self.plan = plan
            self.progress.windows_total = int(plan.get("windows_total") or 0)
            # The anchor frame is the caller's own mask, so it is not produced.
            self.progress.frames_total = int(plan.get("frames_to_produce") or 0)
            self._touch()

    def set_window(self, window_index: int, windows_total: int = 0) -> None:
        """Mark which window is being propagated (1-based).

        Takes the total as well because that is the callback shape the propagator
        reports with, so the job can be wired straight to it.
        """
        with self._lock:
            self.progress.window_index = window_index
            if windows_total:
                self.progress.windows_total = windows_total
            self._touch()

    def add_mask(self, frame_index: int, mask: np.ndarray) -> None:
        """Publish one frame's mask; callers can draw it immediately.

        A frame can be published more than once: when two windows cover it, the
        better-placed one wins and replaces the earlier value. Progress counts
        *distinct* frames, so a re-publish does not push the bar past the total.
        """
        with self._lock:
            if frame_index not in self._masks:
                self.progress.frames_done += 1
            self._masks[frame_index] = mask
            self._touch()

    def add_masks(self, masks: Dict[int, np.ndarray]) -> None:
        """Publish several masks at once (windowing produces them in batches)."""
        if not masks:
            return
        with self._lock:
            for frame_index, mask in masks.items():
                if frame_index not in self._masks:
                    self.progress.frames_done += 1
                self._masks[frame_index] = mask
            self._touch()

    # ── caller-facing API (read from request threads) ───────────────────

    def snapshot(
        self,
        *,
        since: Optional[int] = None,
        until: Optional[int] = None,
    ) -> Tuple[Dict[int, np.ndarray], JobProgress]:
        """A consistent copy of the masks and progress so far.

        `since` returns only frames above that index and `until` only frames below
        it. Both are needed: propagation walks *outward* from the anchor, so a run
        that goes both ways produces frames below everything already sent as well
        as above it, and a single monotonic cursor would silently drop the second
        half. Either way the payload stays proportional to what is new rather than
        to the whole run.
        """
        with self._lock:
            if since is None and until is None:
                masks = dict(self._masks)
            else:
                masks = {
                    frame: mask
                    for frame, mask in self._masks.items()
                    if (since is not None and frame > since)
                    or (until is not None and frame < until)
                }
            progress = JobProgress(**self.progress.as_dict())
            return masks, progress

    def summary(self) -> Dict[str, Any]:
        """Everything about the job except the masks themselves."""
        with self._lock:
            return {
                "job_id": self.id,
                "session_id": self.session_id,
                "object_id": self.object_id,
                "state": self.state.value,
                "error": self.error,
                "anchor": self.anchor,
                "direction": self.direction,
                "first": self.first,
                "last": self.last,
                "progress": self.progress.as_dict(),
                "plan": self.plan,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "updated_at": self.updated_at,
            }

    def _touch(self) -> None:
        self.updated_at = time.time()

    # ── lifecycle (worker thread) ───────────────────────────────────────

    def _start(self, clock: Callable[[], float]) -> None:
        with self._lock:
            self.state = JobState.RUNNING
            self.started_at = clock()
            self._touch()

    def _finish(
        self, state: JobState, error: Optional[str], clock: Callable[[], float]
    ) -> None:
        with self._lock:
            self.state = state
            self.error = error
            self.finished_at = clock()
            self._touch()

    def request_cancel(self, clock: Callable[[], float]) -> None:
        """Ask the job to stop; a queued job stops before it ever starts."""
        with self._lock:
            self.cancel.set()
            self._touch()
            if self.state is JobState.QUEUED:
                self.state = JobState.CANCELLED
                self.finished_at = clock()


#: A runner receives the job and is expected to publish masks through it.
JobRunner = Callable[[PropagationJob], None]


class JobRegistry:
    """A small FIFO queue of long-running jobs, run one at a time.

    Finished jobs are kept for `ttl_seconds` so the reviewer can still read the
    result of the run it just watched, and are then swept away with the session
    they belonged to.
    """

    def __init__(
        self,
        runner: JobRunner,
        *,
        max_jobs: int = 32,
        max_jobs_per_client: int = DEFAULT_PROPAGATE_MAX_JOBS_PER_CLIENT,
        ttl_seconds: int = 1800,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._runner = runner
        self._max_jobs = max(1, max_jobs)
        # Never larger than the queue itself: an allowance above the global
        # ceiling permits nothing extra and would only mislead.
        self._max_jobs_per_client = min(max(1, max_jobs_per_client), self._max_jobs)
        self._ttl_seconds = max(1, ttl_seconds)
        self._clock = clock
        self._jobs: Dict[str, PropagationJob] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[PropagationJob]]" = queue.Queue()
        self._stopping = threading.Event()
        self._worker = threading.Thread(
            target=self._run_forever, name="propagation-jobs", daemon=True
        )

    # ── public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the worker thread (idempotent).

        Guarded by the registry lock: `submit()` is called from request threads,
        so two requests arriving together must not each spawn a worker.
        """
        with self._lock:
            if self._worker.is_alive():
                return
            self._stopping.clear()
            self._worker = threading.Thread(
                target=self._run_forever, name="propagation-jobs", daemon=True
            )
            self._worker.start()

    def submit(
        self,
        *,
        session_id: str,
        client_id: str,
        anchor: int,
        direction: str,
        first: int,
        last: int,
        anchor_mask: Optional[np.ndarray] = None,
        frame_count: int = 0,
        object_id: Optional[int] = None,
        pins: Optional[Dict[int, np.ndarray]] = None,
        chaining: Optional[str] = None,
    ) -> PropagationJob:
        """Queue a propagation run and return it immediately."""
        self.start()
        now = self._clock()
        job = PropagationJob(
            id=str(uuid.uuid4()),
            session_id=session_id,
            client_id=client_id,
            anchor=anchor,
            direction=direction,
            first=first,
            last=last,
            object_id=object_id,
            anchor_mask=anchor_mask,
            frame_count=frame_count,
            pins=dict(pins or {}),
            chaining=chaining,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._evict_locked()
            # The client's own quota is checked first, so a reviewer who filled
            # their share is told exactly that rather than being blamed on the
            # shared queue being full.
            mine = self._live_locked(client_id)
            if mine >= self._max_jobs_per_client:
                raise InvalidRequest(
                    f"You already have {mine} propagation job(s) in flight; the "
                    f"limit is {self._max_jobs_per_client} per client. Wait for one "
                    "to finish, or cancel one."
                )
            in_flight = self._live_locked()
            if in_flight >= self._max_jobs:
                raise InvalidRequest(
                    f"Too many propagation jobs in flight ({in_flight}, the limit "
                    f"is {self._max_jobs}) across all clients; wait for one to "
                    "finish or close the clip."
                )
            self._jobs[job.id] = job
            self._order.append(job.id)
        # A provisional total: the runner replaces it with the plan's own count,
        # which also subtracts the verified frames the run will not produce.
        job.progress.frames_total = max(0, last - first - len(job.pins))
        self._queue.put(job)
        return job

    def get(self, job_id: str, *, client_id: str) -> PropagationJob:
        """One client's job, or ``NotFound``.

        A job that belongs to someone else is reported as unknown rather than as
        forbidden, so a caller cannot probe for other clients' work.
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.client_id != client_id:
            raise NotFound(f"Propagation job {job_id} is unknown or has expired.")
        return job

    def cancel(self, job_id: str, *, client_id: str) -> PropagationJob:
        job = self.get(job_id, client_id=client_id)
        job.request_cancel(self._clock)
        return job

    def list(self, *, client_id: str) -> List[PropagationJob]:
        """This client's jobs, oldest first (the queue order the reviewer shows)."""
        with self._lock:
            self._evict_locked()
            return [
                self._jobs[job_id]
                for job_id in self._order
                if job_id in self._jobs and self._jobs[job_id].client_id == client_id
            ]

    def _all_locked(self) -> List[PropagationJob]:
        """Every job, whatever its owner. The caller holds the lock.

        The registry's own bookkeeping (eviction, shutdown) needs the whole set;
        only the client-facing `list` is scoped.
        """
        return [self._jobs[job_id] for job_id in self._order if job_id in self._jobs]

    def queued_count(self) -> int:
        with self._lock:
            return self._queued_locked()

    def counts(self) -> Dict[str, int]:
        """How many jobs are waiting and how many are running, across all clients.

        Global on purpose: it describes the one GPU everybody shares, so it is the
        number that explains why a job just submitted has not started moving.
        """
        with self._lock:
            self._evict_locked()
            jobs = self._all_locked()
        return {
            "queued": sum(1 for job in jobs if job.state is JobState.QUEUED),
            "running": sum(1 for job in jobs if job.state is JobState.RUNNING),
        }

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the worker, cancelling whatever it is doing."""
        with self._lock:
            self._evict_locked()
            known = self._all_locked()
        for job in known:
            if not job.state.finished:
                job.request_cancel(self._clock)
        self._stopping.set()
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=timeout)
        with self._lock:
            self._jobs.clear()
            self._order.clear()

    # ── internals ───────────────────────────────────────────────────────

    def _queued_locked(self) -> int:
        return sum(1 for job in self._jobs.values() if job.state is JobState.QUEUED)

    def _live_locked(self, client_id: Optional[str] = None) -> int:
        """Jobs still queued or running — the ones holding a place in the queue.

        Finished jobs are deliberately *not* counted, even though they are retained
        so the reviewer can read the result. Counting them would mean a client who
        once ran their share of jobs was locked out until the TTL expired, and the
        queue would start refusing work because of history rather than load.
        """
        return sum(
            1
            for job in self._jobs.values()
            if not job.state.finished
            and (client_id is None or job.client_id == client_id)
        )

    def _evict_locked(self) -> None:
        """Drop finished jobs that are past their TTL. Caller holds the lock."""
        now = self._clock()
        keep: List[str] = []
        for job_id in self._order:
            job = self._jobs.get(job_id)
            if job is None:
                continue
            expired = (
                job.state.finished
                and job.finished_at is not None
                and now - job.finished_at > self._ttl_seconds
            )
            if expired:
                self._jobs.pop(job_id, None)
            else:
                keep.append(job_id)
        self._order = keep

    def _run_forever(self) -> None:
        while True:
            job = self._queue.get()
            if job is None or self._stopping.is_set():
                return
            if job.state is JobState.CANCELLED:
                continue  # cancelled while queued
            job._start(self._clock)
            try:
                self._runner(job)
            except JobCancelled:
                job._finish(JobState.CANCELLED, None, self._clock)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - the job must never kill the worker
                job._finish(
                    JobState.FAILED, f"{type(exc).__name__}: {exc}", self._clock
                )
            else:
                state = JobState.CANCELLED if job.cancel.is_set() else JobState.DONE
                job._finish(state, None, self._clock)

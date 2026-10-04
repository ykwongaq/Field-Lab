"""HTTP checks for the `/api/propagate` status readout.

Only status is covered here. The job endpoints need a real propagator and a real
clip, and their behaviour — scoping, planning, publishing masks — is exercised in
`test_propagation.py` against a stub predictor, which is where it belongs.

What this file adds is the one thing unit tests cannot show: that the numbers a
reviewer sees describe the *live* registry rather than a constant.
"""

from __future__ import annotations

import threading

from src.core.jobs import JobRegistry
from src.domain.windows import DIRECTION_FORWARD
from src.inference.registry import get_job_registry
from src.main import app


def test_status_reports_an_idle_queue(api_client):
    """A quiet server says so, and still reports the window sizing."""
    body = api_client.get("/api/propagate/status").json()
    assert body["queued_jobs"] == 0, body
    assert body["running_jobs"] == 0, body
    assert body["gpu_busy"] is False, body
    assert body["gpu_waiting"] == 0, body
    assert body["window_frames"] >= 2, body
    assert body["chaining"] in ("derived", "verified"), body


def test_status_counts_jobs_in_flight(api_client):
    """The readout comes from the live registry, not from constants.

    This is the number that explains why a run which was just accepted has not
    started moving yet, so it is worth proving it is actually wired up.
    """
    release = threading.Event()
    started = threading.Event()

    def runner(_job):
        started.set()
        release.wait(timeout=10)

    registry = JobRegistry(runner, max_jobs=8, max_jobs_per_client=4, ttl_seconds=1800)
    app.dependency_overrides[get_job_registry] = lambda: registry
    try:
        for _ in range(2):
            registry.submit(
                session_id="s-status",
                client_id="a" * 32,
                anchor=0,
                direction=DIRECTION_FORWARD,
                first=0,
                last=4,
            )
        assert started.wait(timeout=5), "the worker never picked a job up"

        body = api_client.get("/api/propagate/status").json()
        assert body["running_jobs"] == 1, body
        assert body["queued_jobs"] == 1, body
    finally:
        app.dependency_overrides.pop(get_job_registry, None)
        release.set()
        registry.shutdown()

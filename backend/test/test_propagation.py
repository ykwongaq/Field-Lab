"""Checks for windowed mask propagation, driven by a stub predictor.

The point of the stub is that the *request sequence* — start a session per
window, condition it with masks from the previous window, propagate with the
tracker forced, close the session — is what this code is responsible for. The
model itself is not, so it does not appear here. Nothing in this file needs a
GPU.

The stub's masks encode which window produced them (a single pixel in the
column of the window-local frame index), which makes the stitching rule
observable.
"""

import os
import threading
import time

import numpy as np
import pytest
from PIL import Image

from src.core.errors import InvalidRequest, NotFound
from src.core.jobs import JobRegistry
from src.core.sessions import create_session, frame_name
from src.domain.windows import DIRECTION_BACKWARD, DIRECTION_FORWARD
from src.inference.sam3_video import (
    PropagateConfig,
    PropagateError,
    PropagateUnavailable,
    Sam3VideoPropagator,
)

HEIGHT, WIDTH = 6, 24

#: Every session and job belongs to a client. These tests act as one client; the
#: registry-scoping check below uses a second. See `core.identity`.
OWNER = "a" * 32


@pytest.fixture
def make_session(scratch):
    """`make_session(frame_count, tag)` -> a throwaway session in the test's dir."""

    def make(frame_count: int, tag: str):
        """A session holding `frame_count` tiny JPEGs."""
        session = create_session(scratch, owner=OWNER)
        for index in range(frame_count):
            Image.new("RGB", (WIDTH, HEIGHT), (index * 7 % 255, 60, 120)).save(
                os.path.join(session.frames_dir, frame_name(index)), format="JPEG"
            )
        session.update_meta(tag=tag)
        return session

    return make


def marker(local, shape=(HEIGHT, WIDTH)):
    """A mask that says 'produced by the window where this frame was local N'."""
    mask = np.zeros(shape, dtype=bool)
    if 0 <= local < shape[1]:
        mask[0, local] = True
    return mask


class StubPredictor:
    """A stand-in for `Sam3VideoPredictor`: records the conversation."""

    def __init__(self, *, shape=(HEIGHT, WIDTH), on_yield=None, fail_on_start=False):
        self.shape = shape
        self.on_yield = on_yield
        self.fail_on_start = fail_on_start
        self.sessions = {}
        self.requests = []
        self.streams = []
        self.add_masks = []
        self.closed = []

    # -- the two entry points the propagator uses --------------------------

    def handle_request(self, request):
        self.requests.append(dict(request))
        kind = request["type"]
        if kind == "start_session":
            if self.fail_on_start:
                raise RuntimeError("CUDA out of memory")
            session_id = f"s{len(self.sessions)}"
            self.sessions[session_id] = list(request["resource_path"])
            return {"session_id": session_id}
        if kind == "add_mask":
            self.add_masks.append(dict(request))
            return {"frame_index": request["frame_index"], "outputs": {}}
        if kind == "close_session":
            self.closed.append(request["session_id"])
            return {"is_success": True}
        raise AssertionError(f"unexpected request type {kind!r}")

    def handle_stream_request(self, request):
        self.streams.append(dict(request))
        frames = self.sessions[request["session_id"]]
        for local in range(len(frames)):
            if self.on_yield is not None:
                self.on_yield(local)
            yield {
                "frame_index": local,
                "outputs": {
                    "out_obj_ids": np.array([1]),
                    "out_binary_masks": np.array([marker(local, self.shape)]),
                    "out_probs": np.array([0.9]),
                },
            }


def propagator(predictor, **config):
    settings = {
        "window_frames": 10,
        "overlap": 4,
        "anchor_max": 3,
    }
    settings.update(config)
    return Sam3VideoPropagator(
        PropagateConfig(**settings), predictor_provider=lambda: predictor
    )


def run(session, predictor, *, anchor, mask, direction=DIRECTION_FORWARD, **kwargs):
    plan = kwargs.pop("plan", None)
    engine = kwargs.pop("engine", None) or propagator(predictor)
    if plan is None:
        last = kwargs.pop("last", session.frame_count() - 1)
        first = kwargs.pop("first", 0)
        plan = engine.plan(
            anchor=anchor,
            first=first,
            last=last,
            direction=direction,
            frame_count=session.frame_count(),
        )
    return (
        engine,
        plan,
        engine.propagate(session, anchor=anchor, anchor_mask=mask, plan=plan, **kwargs),
    )


ANCHOR_MASK = np.zeros((HEIGHT, WIDTH), dtype=bool)
ANCHOR_MASK[2:4, 2:4] = True


def test_windows_are_used_and_everything_is_covered(make_session):
    session = make_session(16, "coverage")
    predictor = StubPredictor()
    engine, plan, produced = run(
        session, predictor, anchor=0, mask=ANCHOR_MASK, direction=DIRECTION_FORWARD
    )
    assert plan.windows_total == 2, f"expected 2 windows, got {plan.windows_total}"
    assert len(predictor.sessions) == 2, "one session per window"
    assert sorted(predictor.closed) == ["s0", "s1"], "every session is closed"
    assert set(produced) == set(range(0, 16)), f"coverage: {sorted(produced)}"


def test_stitching_prefers_the_most_interior_window(make_session):
    """Frame 6 and frame 8 are covered twice; the better-placed window wins."""
    session = make_session(16, "stitch")
    predictor = StubPredictor()
    engine, plan, produced = run(
        session, predictor, anchor=0, mask=ANCHOR_MASK, direction=DIRECTION_FORWARD
    )
    # Window 1 is [0, 10) and window 2 is [6, 16).
    assert (
        np.flatnonzero(produced[6][0])[0] == 6
    ), "frame 6 belongs to window 1 (score 3) rather than window 2 (score 0)"
    assert (
        np.flatnonzero(produced[8][0])[0] == 2
    ), "frame 8 belongs to window 2 (score 2) rather than window 1 (score 1)"


def test_the_anchor_frame_is_never_overwritten(make_session):
    session = make_session(12, "anchor")
    predictor = StubPredictor()
    engine, plan, produced = run(
        session, predictor, anchor=5, mask=ANCHOR_MASK, direction=DIRECTION_FORWARD
    )
    assert np.array_equal(
        produced[5], ANCHOR_MASK
    ), "the caller's mask is authoritative on its own frame"


def test_later_windows_are_conditioned_on_the_overlap(make_session):
    session = make_session(16, "anchors")
    predictor = StubPredictor()
    run(session, predictor, anchor=0, mask=ANCHOR_MASK, direction=DIRECTION_FORWARD)

    by_session = {}
    for request in predictor.add_masks:
        by_session.setdefault(request["session_id"], []).append(request["frame_index"])
    assert by_session.get("s0") == [0], "window 1 is anchored on the caller's frame"
    # Window 2 shares frames 6..9; the planner spreads three anchors across them.
    assert by_session.get("s1") == [0, 2, 3], by_session.get("s1")
    assert all(
        request["obj_id"] == 1 for request in predictor.add_masks
    ), "one object per window"
    assert np.asarray(predictor.add_masks[0]["mask"]).shape == (HEIGHT, WIDTH)


def test_tracking_is_forced_and_follows_the_window_direction(make_session):
    session = make_session(16, "force")
    predictor = StubPredictor()
    _, plan, _ = run(
        session, predictor, anchor=0, mask=ANCHOR_MASK, direction=DIRECTION_FORWARD
    )
    assert len(predictor.streams) == plan.windows_total
    for request in predictor.streams:
        assert request["force_tracker_propagation"] is True, (
            "add_mask records a refinement; without forcing, the tracker would "
            "replay cached predictions"
        )
    assert predictor.streams[0]["propagation_direction"] == DIRECTION_FORWARD


def test_progress_callbacks_report_frames_and_windows(make_session):
    session = make_session(16, "progress")
    predictor = StubPredictor()
    events = []
    windows = []
    current = {"window": 0}
    engine, plan, produced = run(
        session,
        predictor,
        anchor=0,
        mask=ANCHOR_MASK,
        direction=DIRECTION_FORWARD,
        on_mask=lambda index, mask: events.append((current["window"], index)),
        on_window=lambda index, total: (
            windows.append((index, total)),
            current.update(window=index),
        ),
    )
    assert windows == [(1, 2), (2, 2)], windows
    assert events, "masks must be reported as they are produced"
    # Each window publishes its own frames in ascending order. A frame that two
    # windows cover can be published twice on purpose: the better-placed window
    # replaces the earlier value, which is what makes the overlay settle.
    for window_index in (1, 2):
        published = [frame for window, frame in events if window == window_index]
        window = plan.windows[window_index - 1]
        assert published == sorted(published), f"window {window_index} is not in order"
        assert all(window.start <= frame < window.end for frame in published), published
    # A later window only speaks for the frames it wins, so the second window
    # starts at the first overlap frame it improves on (not at its own start).
    assert 8 in [frame for window, frame in events if window == 2], events
    assert 0 not in [
        frame for _window, frame in events
    ], "the anchor frame is the caller's mask, so it is never re-published"
    assert {frame for _window, frame in events} == set(produced) - {0}, (
        "every propagated frame was published at least once (the anchor is not "
        "published: it is the caller's mask, not a prediction)"
    )


def test_cancelling_stops_the_run_and_keeps_what_was_produced(make_session):
    session = make_session(40, "cancel")
    cancel = threading.Event()
    state = {"session": 0}

    def on_yield(local):
        # Cancel once the second window is under way.
        if state["session"] >= 1 and local >= 2:
            cancel.set()

    def on_window(_index, _total):
        state["session"] += 1

    predictor = StubPredictor(on_yield=on_yield)
    engine, plan, produced = run(
        session,
        predictor,
        anchor=0,
        mask=ANCHOR_MASK,
        direction=DIRECTION_FORWARD,
        cancel=cancel,
        on_window=on_window,
    )
    assert plan.windows_total > 2, "the test needs several windows"
    assert len(predictor.streams) < plan.windows_total, "the run stopped early"
    assert produced, "the frames produced before the cancel are kept"
    assert len(predictor.closed) == len(predictor.sessions), "sessions still close"


def test_backward_runs_are_planned_the_other_way(make_session):
    session = make_session(16, "backward")
    predictor = StubPredictor()
    engine, plan, produced = run(
        session,
        predictor,
        anchor=15,
        mask=ANCHOR_MASK,
        direction=DIRECTION_BACKWARD,
        first=0,
        last=15,
    )
    assert plan.windows[0].end == 16, "the first window ends just past the anchor"
    assert set(produced) == set(range(0, 16)), "coverage"
    assert predictor.streams[0]["propagation_direction"] == DIRECTION_BACKWARD


def test_anchor_mask_shape(make_session):
    """A leading singleton channel is fine; anything else is not a mask."""
    session = make_session(4, "shape")
    predictor = StubPredictor()
    engine = propagator(predictor)
    plan = engine.plan(
        anchor=0, first=0, last=3, direction=DIRECTION_FORWARD, frame_count=4
    )
    produced = engine.propagate(
        session,
        anchor=0,
        anchor_mask=ANCHOR_MASK[None, ...],  # (1, H, W), as a model would return it
        plan=plan,
    )
    assert np.array_equal(produced[0], ANCHOR_MASK), "a (1, H, W) mask is squeezed"
    with pytest.raises(PropagateError, match="2-D"):
        engine.propagate(
            session, anchor=0, anchor_mask=np.zeros((2, 2, 2), dtype=bool), plan=plan
        )


def test_predictor_failures_are_not_swallowed(make_session):
    """A failed window surfaces as an error the caller can act on."""
    session = make_session(4, "failure")
    predictor = StubPredictor(fail_on_start=True)
    engine = propagator(predictor)
    plan = engine.plan(
        anchor=0, first=0, last=3, direction=DIRECTION_FORWARD, frame_count=4
    )
    with pytest.raises(PropagateUnavailable, match="out of memory"):
        engine.propagate(session, anchor=0, anchor_mask=ANCHOR_MASK, plan=plan)


def test_missing_frames_are_reported(make_session):
    """A frame missing from the session is a 404, not a crash."""
    session = make_session(4, "missing")
    os.remove(os.path.join(session.frames_dir, frame_name(2)))
    predictor = StubPredictor()
    engine = propagator(predictor)
    plan = engine.plan(
        anchor=0, first=0, last=3, direction=DIRECTION_FORWARD, frame_count=4
    )
    with pytest.raises(NotFound):
        engine.propagate(session, anchor=0, anchor_mask=ANCHOR_MASK, plan=plan)


def test_propagation_runner_resolves_the_session_and_plans(make_session, scratch):
    """The runner is what the job queue calls; it must plan and publish."""
    from src.core.config import Settings
    from src.core.jobs import PropagationJob
    from src.inference.sam3_video import PropagationRunner

    session = make_session(12, "runner")
    predictor = StubPredictor()
    engine = propagator(predictor)
    job = PropagationJob(
        id="job-1",
        session_id=session.id,
        client_id=OWNER,
        anchor=0,
        direction=DIRECTION_FORWARD,
        first=0,
        last=11,
        anchor_mask=ANCHOR_MASK,
        frame_count=12,
    )
    PropagationRunner(engine, Settings(temp_dir=scratch))(job)
    masks, progress = job.snapshot()
    # The anchor frame is the caller's own mask, so the run does not produce it,
    # and progress is counted against the frames that do have to be produced.
    assert set(masks) == set(range(1, 12)), f"runner coverage: {sorted(masks)}"
    assert progress.frames_done == 11, progress.as_dict()
    assert progress.frames_total == 11, progress.as_dict()
    assert job.plan is not None and job.plan["windows_total"] == 2, job.plan
    assert job.plan["frames_to_produce"] == 11, job.plan
    assert "elapsed_ms" in job.plan, "the plan records how long the run took"


def test_verified_frames_anchor_and_are_never_overwritten(make_session):
    """A pin is authoritative, exactly like the anchor. The tracker cannot replace it."""
    session = make_session(16, "pins")
    predictor = StubPredictor()
    pin = np.zeros((HEIGHT, WIDTH), dtype=bool)
    pin[5, 5] = True

    engine, plan, produced = run(
        session,
        predictor,
        anchor=0,
        mask=ANCHOR_MASK,
        direction=DIRECTION_FORWARD,
        pins={8: pin},
    )
    by_session = {}
    for request in predictor.add_masks:
        by_session.setdefault(request["session_id"], []).append(request["frame_index"])

    # Window 1 is [0, 10): it holds the anchor (frame 0) and the pin (frame 8).
    assert by_session.get("s0") == [0, 8], by_session.get("s0")
    # Window 2 is [6, 16): the anchor is behind it, so the pin is the only thing it
    # may be conditioned on. Frames 6, 7 and 9 carry window 1's own predictions and
    # are deliberately not promoted to anchors.
    assert by_session.get("s1") == [2], by_session.get("s1")
    assert np.array_equal(
        produced[8], pin
    ), "the verified mask outranks the prediction on its own frame"
    assert np.array_equal(produced[0], ANCHOR_MASK)
    assert set(produced) == set(range(0, 16)), "coverage is unchanged"


def test_strict_chaining_stops_where_verification_ends(make_session):
    """`verified` refuses a derived hand-off, so the run stops at the window edge."""
    session = make_session(16, "strict")
    predictor = StubPredictor()
    engine, plan, produced = run(
        session,
        predictor,
        anchor=0,
        mask=ANCHOR_MASK,
        direction=DIRECTION_FORWARD,
        chaining="verified",
    )
    assert plan.windows_total == 2, "the plan still describes the whole range"
    assert len(predictor.sessions) == 1, "only the verified window ran"
    assert len(predictor.streams) == 1, "no chain stepped past the verification"
    assert set(produced) == set(range(0, 10)), sorted(produced)
    assert len(predictor.closed) == 1, "the session still closes"


def test_an_unknown_chaining_mode_is_rejected():
    """`chaining` is `derived` or `verified`, and derived is the default."""
    with pytest.raises(ValueError, match="chaining"):
        PropagateConfig(chaining="sometimes")
    assert PropagateConfig().allow_derived is True, "derived is the default"
    assert PropagateConfig(chaining="verified").allow_derived is False


def test_the_runner_counts_verified_frames_out_of_progress(make_session, scratch):
    """Pins are input, not output: they must not leave the progress bar short."""
    from src.core.config import Settings
    from src.core.jobs import PropagationJob
    from src.inference.sam3_video import PropagationRunner

    session = make_session(12, "pin-progress")
    predictor = StubPredictor()
    engine = propagator(predictor)
    pin = np.zeros((HEIGHT, WIDTH), dtype=bool)
    pin[1, 1] = True
    job = PropagationJob(
        id="job-pin-progress",
        session_id=session.id,
        client_id=OWNER,
        anchor=0,
        direction=DIRECTION_FORWARD,
        first=0,
        last=11,
        anchor_mask=ANCHOR_MASK,
        frame_count=12,
        pins={4: pin},
    )
    PropagationRunner(engine, Settings(temp_dir=scratch))(job)
    masks, progress = job.snapshot()
    # A job publishes the frames it *produced*. The anchor and the verified frame
    # are input — the caller already holds both — so they are absent here, which is
    # also why they must not be counted as outstanding work.
    assert set(masks) == set(range(0, 12)) - {0, 4}, sorted(masks)
    assert progress.frames_total == 10, progress.as_dict()
    assert progress.frames_done == 10, progress.as_dict()
    assert job.plan["frames_to_produce"] == 10, job.plan
    assert job.plan["frames_produced"] == 10, job.plan
    assert job.plan["pinned"] == 1, job.plan


def test_a_verified_frame_the_run_never_visits_is_rejected(make_session, scratch):
    """A pin outside the planned range is an error, not a pin that does nothing."""
    from src.core.config import Settings
    from src.core.jobs import PropagationJob
    from src.inference.sam3_video import PropagationRunner

    session = make_session(16, "pin-range")
    predictor = StubPredictor()
    engine = propagator(predictor)
    job = PropagationJob(
        id="job-pin-range",
        session_id=session.id,
        client_id=OWNER,
        anchor=6,
        direction=DIRECTION_FORWARD,
        first=0,
        last=11,
        anchor_mask=ANCHOR_MASK,
        frame_count=16,
        pins={3: np.ones((HEIGHT, WIDTH), dtype=bool)},
    )
    with pytest.raises(PropagateError, match="outside the frames"):
        PropagationRunner(engine, Settings(temp_dir=scratch))(job)
    assert predictor.sessions == {}, "nothing was loaded before the request was refused"


def test_pin_validation_rejects_unusable_frames():
    """Shape, emptiness, duplicates and the anchor frame are all refused up front."""
    from src.api.routes.propagate import _decode_pins
    from src.domain.rle import encode_rle
    from src.schemas.propagate import PinnedMask

    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[3, 3] = True
    size, counts = encode_rle(mask)
    pin = PinnedMask(frame_index=4, mask={"size": size, "counts": counts})

    decoded = _decode_pins([pin], anchor=0, width=WIDTH, height=HEIGHT)
    assert set(decoded) == {4} and np.array_equal(decoded[4], mask)

    empty = encode_rle(np.zeros((HEIGHT, WIDTH), dtype=bool))[1]
    cases = [
        (
            [PinnedMask(frame_index=0, mask={"size": size, "counts": counts})],
            WIDTH,
            "anchor frame",
        ),
        ([pin, pin], WIDTH, "more than once"),
        (
            [PinnedMask(frame_index=9, mask={"size": size, "counts": empty})],
            WIDTH,
            "is empty",
        ),
        ([pin], WIDTH + 4, "but the frames are"),
    ]
    # Named cases rather than four copies of the same try/except, and each asserts
    # *which* rejection it got: an empty pin and a wrong-sized one used to be
    # indistinguishable from any other refusal.
    for pins, width, reason in cases:
        with pytest.raises(InvalidRequest, match=reason):
            _decode_pins(pins, anchor=0, width=width, height=HEIGHT)


def test_the_gate_is_released_between_windows(make_session):
    """A prompt must get in while a run is between windows, not only after it.

    The run holds the model gate per window rather than for its whole length,
    which is what stops one long propagation from blocking every other user's
    click until the clip is finished.
    """
    session = make_session(24, "gate")
    predictor = StubPredictor()
    engine = propagator(predictor)
    gate = engine._manager.gate
    busy_between_windows = []

    def on_window(_index, _total):
        busy_between_windows.append(gate.busy)

    _engine, plan, _produced = run(
        session,
        predictor,
        anchor=0,
        mask=ANCHOR_MASK,
        engine=engine,
        on_window=on_window,
    )
    assert plan.windows_total > 1, "the test needs several windows"
    assert len(busy_between_windows) == plan.windows_total, busy_between_windows
    assert not any(busy_between_windows), (
        "the gate was still held between windows, so a prompt would have queued "
        "behind the whole run"
    )


def test_the_job_registry_scopes_reads_to_the_owner():
    """A job is invisible to a client that did not submit it.

    The queue is global — one GPU runs one job at a time — but a client may only
    see, poll and cancel its own runs, or the job list becomes a way to read
    someone else's masks and stop their work.
    """
    alice, bob = "a" * 32, "b" * 32
    registry = JobRegistry(lambda _job: None, clock=lambda: 0.0)
    mine = registry.submit(
        session_id="s-mine",
        client_id=alice,
        anchor=0,
        direction=DIRECTION_FORWARD,
        first=0,
        last=4,
    )
    theirs = registry.submit(
        session_id="s-theirs",
        client_id=bob,
        anchor=0,
        direction=DIRECTION_FORWARD,
        first=0,
        last=4,
    )
    try:
        assert [job.id for job in registry.list(client_id=alice)] == [mine.id]
        assert [job.id for job in registry.list(client_id=bob)] == [theirs.id]
        assert registry.get(mine.id, client_id=alice).id == mine.id

        # Both a different client and an empty one: a job must not become readable
        # just because the caller stopped claiming an identity.
        for intruder in (bob, ""):
            with pytest.raises(NotFound):
                registry.get(mine.id, client_id=intruder)
            with pytest.raises(NotFound):
                registry.cancel(mine.id, client_id=intruder)
    finally:
        registry.shutdown()


class _Hold:
    """A runner that parks every job until released.

    The queue caps only count jobs that are *in flight*, so a runner that returns
    immediately leaves nothing for them to observe. This holds each job in the
    worker until the test says otherwise.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, _job) -> None:
        self.entered.set()
        self.release.wait(timeout=10)


def submit(registry, client_id):
    """One placeholder propagation job for `client_id`."""
    return registry.submit(
        session_id="s-in-flight",
        client_id=client_id,
        anchor=0,
        direction=DIRECTION_FORWARD,
        first=0,
        last=4,
    )


def wait_finished(job, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not job.state.finished:
        if time.monotonic() > deadline:
            raise AssertionError(f"job {job.id} never finished")
        time.sleep(0.01)


def test_one_client_cannot_fill_the_queue():
    """A reviewer is held to their own share, however empty the queue is.

    The global ceiling alone lets one client take every slot, so every other
    reviewer is told the queue is full while the GPU sits idle behind work that
    only that one client asked for.
    """
    alice, bob = "a" * 32, "b" * 32
    hold = _Hold()
    registry = JobRegistry(
        hold, max_jobs=6, max_jobs_per_client=2, ttl_seconds=1800
    )
    try:
        submit(registry, alice)
        submit(registry, alice)
        with pytest.raises(InvalidRequest, match="per client"):
            submit(registry, alice)

        # Bob's own share is untouched by Alice filling hers.
        submit(registry, bob)
        assert registry.counts()["running"] + registry.counts()["queued"] == 3
    finally:
        hold.release.set()
        registry.shutdown()


def test_a_finished_job_does_not_hold_a_place():
    """Retained results are history, not capacity.

    A finished run is kept for the TTL so the reviewer can still read it. If that
    counted against the caps, a client who used their share once could not start
    another run until the TTL expired, and the queue would refuse work because of
    history rather than load.
    """
    alice = "a" * 32
    registry = JobRegistry(
        lambda _job: None, max_jobs=2, max_jobs_per_client=1, ttl_seconds=1800
    )
    try:
        for _ in range(3):
            wait_finished(submit(registry, alice))
    finally:
        registry.shutdown()


def test_the_registry_reports_the_shared_queue():
    """`counts()` describes the one GPU everybody shares, in flight only."""
    hold = _Hold()
    registry = JobRegistry(
        hold, max_jobs=8, max_jobs_per_client=4, ttl_seconds=1800
    )
    try:
        assert registry.counts() == {"queued": 0, "running": 0}

        submit(registry, "a" * 32)
        assert hold.entered.wait(timeout=5), "the worker never picked the job up"
        submit(registry, "a" * 32)
        assert registry.counts() == {"queued": 1, "running": 1}
    finally:
        hold.release.set()
        registry.shutdown()

"""Checks for windowed mask propagation, driven by a stub predictor.

Run from the backend root::

    python test/test_propagation.py

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
import shutil
import sys
import tempfile
import threading

import numpy as np
from PIL import Image

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from src.core.errors import NotFound  # noqa: E402
from src.core.sessions import create_session, frame_name  # noqa: E402
from src.domain.windows import DIRECTION_BACKWARD, DIRECTION_FORWARD  # noqa: E402
from src.inference.sam3_video import (  # noqa: E402
    PropagateConfig,
    Sam3VideoPropagator,
)

checks = 0
ROOT = tempfile.mkdtemp(prefix="vsr-propagate-")
HEIGHT, WIDTH = 6, 24


def ok(label):
    global checks
    checks += 1
    print("  ok:", label)


def make_session(frame_count, tag):
    """A throwaway session holding `frame_count` tiny JPEGs."""
    session = create_session(ROOT)
    for index in range(frame_count):
        Image.new("RGB", (WIDTH, HEIGHT), (index * 7 % 255, 60, 120)).save(
            os.path.join(session.frames_dir, frame_name(index)), format="JPEG"
        )
    session.update_meta(tag=tag)
    return session


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


def test_windows_are_used_and_everything_is_covered():
    session = make_session(16, "coverage")
    predictor = StubPredictor()
    engine, plan, produced = run(
        session, predictor, anchor=0, mask=ANCHOR_MASK, direction=DIRECTION_FORWARD
    )
    assert plan.windows_total == 2, f"expected 2 windows, got {plan.windows_total}"
    assert len(predictor.sessions) == 2, "one session per window"
    assert sorted(predictor.closed) == ["s0", "s1"], "every session is closed"
    assert set(produced) == set(range(0, 16)), f"coverage: {sorted(produced)}"
    ok("16 frames in windows of 10/overlap 4: 2 sessions, every frame produced")


def test_stitching_prefers_the_most_interior_window():
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
    ok("overlapping frames keep the value from the more interior window")


def test_the_anchor_frame_is_never_overwritten():
    session = make_session(12, "anchor")
    predictor = StubPredictor()
    engine, plan, produced = run(
        session, predictor, anchor=5, mask=ANCHOR_MASK, direction=DIRECTION_FORWARD
    )
    assert np.array_equal(
        produced[5], ANCHOR_MASK
    ), "the caller's mask is authoritative on its own frame"
    ok("the anchor frame keeps the caller's mask, not the tracker's")


def test_later_windows_are_conditioned_on_the_overlap():
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
    ok("window 2 is anchored on three masks taken from the overlap")


def test_tracking_is_forced_and_follows_the_window_direction():
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
    ok("every window propagates with the tracker forced, in the window's direction")


def test_progress_callbacks_report_frames_and_windows():
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
    ok(
        f"masks stream out per window ({len(events)} publications) and windows announce themselves"
    )


def test_cancelling_stops_the_run_and_keeps_what_was_produced():
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
    ok(f"cancel stopped after {len(predictor.streams)} of {plan.windows_total} windows")


def test_backward_runs_are_planned_the_other_way():
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
    ok("a backward run starts at the anchor and walks down")


def test_anchor_mask_shape():
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
    try:
        engine.propagate(
            session, anchor=0, anchor_mask=np.zeros((2, 2, 2), dtype=bool), plan=plan
        )
    except Exception as exc:  # PropagateError
        assert "2-D" in str(exc), str(exc)
    else:
        raise AssertionError("a mask that is not 2-D must be rejected")
    ok("a (1, H, W) mask is accepted and a 3-D one is rejected")


def test_predictor_failures_are_not_swallowed():
    session = make_session(4, "failure")
    predictor = StubPredictor(fail_on_start=True)
    engine = propagator(predictor)
    plan = engine.plan(
        anchor=0, first=0, last=3, direction=DIRECTION_FORWARD, frame_count=4
    )
    try:
        engine.propagate(session, anchor=0, anchor_mask=ANCHOR_MASK, plan=plan)
    except Exception as exc:
        assert "out of memory" in str(exc), str(exc)
    else:
        raise AssertionError("a predictor failure must reach the caller")
    ok("a failed window surfaces as an error instead of silence")


def test_missing_frames_are_reported():
    session = make_session(4, "missing")
    os.remove(os.path.join(session.frames_dir, frame_name(2)))
    predictor = StubPredictor()
    engine = propagator(predictor)
    plan = engine.plan(
        anchor=0, first=0, last=3, direction=DIRECTION_FORWARD, frame_count=4
    )
    try:
        engine.propagate(session, anchor=0, anchor_mask=ANCHOR_MASK, plan=plan)
    except NotFound:
        ok("a frame missing from the session is a 404, not a crash")
    else:
        raise AssertionError("a missing frame must be reported")


def test_propagation_runner_resolves_the_session_and_plans():
    """The runner is what the job queue calls; it must plan and publish."""
    from src.core.config import Settings
    from src.core.jobs import PropagationJob
    from src.inference.sam3_video import PropagationRunner

    session = make_session(12, "runner")
    predictor = StubPredictor()

    class Registry:
        """Capture the masks the runner publishes, without a queue."""

        def __init__(self):
            self.masks = {}
            self.plan = None

        def __call__(self, job):
            PropagationRunner(engine, settings)(job)

    engine = propagator(predictor)
    settings = Settings(temp_dir=ROOT)
    job = PropagationJob(
        id="job-1",
        session_id=session.id,
        anchor=0,
        direction=DIRECTION_FORWARD,
        first=0,
        last=11,
        anchor_mask=ANCHOR_MASK,
        frame_count=12,
    )
    PropagationRunner(engine, settings)(job)
    masks, progress = job.snapshot()
    # The anchor frame is the caller's own mask, so the run does not produce it,
    # and progress is counted against the frames that do have to be produced.
    assert set(masks) == set(range(1, 12)), f"runner coverage: {sorted(masks)}"
    assert progress.frames_done == 11, progress.as_dict()
    assert progress.frames_total == 11, progress.as_dict()
    assert job.plan is not None and job.plan["windows_total"] == 2, job.plan
    assert job.plan["frames_to_produce"] == 11, job.plan
    assert "elapsed_ms" in job.plan, "the plan records how long the run took"
    ok("the job runner opens the session, plans, propagates and publishes masks")


def test_verified_frames_anchor_and_are_never_overwritten():
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
    ok("a verified frame anchors its windows and the run cannot overwrite it")


def test_strict_chaining_stops_where_verification_ends():
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
    ok("strict chaining stops instead of seeding a window from a prediction")


def test_an_unknown_chaining_mode_is_rejected():
    try:
        PropagateConfig(chaining="sometimes")
    except ValueError as exc:
        assert "chaining" in str(exc), str(exc)
    else:
        raise AssertionError("an unknown chaining mode must be rejected")
    assert PropagateConfig().allow_derived is True, "derived is the default"
    assert PropagateConfig(chaining="verified").allow_derived is False
    ok("chaining is `derived` or `verified`, and derived is the default")


def test_the_runner_counts_verified_frames_out_of_progress():
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
        anchor=0,
        direction=DIRECTION_FORWARD,
        first=0,
        last=11,
        anchor_mask=ANCHOR_MASK,
        frame_count=12,
        pins={4: pin},
    )
    PropagationRunner(engine, Settings(temp_dir=ROOT))(job)
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
    ok("verified frames are input: progress reaches 100 % once the rest is produced")


def test_a_verified_frame_the_run_never_visits_is_rejected():
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
        anchor=6,
        direction=DIRECTION_FORWARD,
        first=0,
        last=11,
        anchor_mask=ANCHOR_MASK,
        frame_count=16,
        pins={3: np.ones((HEIGHT, WIDTH), dtype=bool)},
    )
    try:
        PropagationRunner(engine, Settings(temp_dir=ROOT))(job)
    except Exception as exc:
        assert "outside the frames" in str(exc), str(exc)
    else:
        raise AssertionError("a pin the run cannot visit must be an error")
    assert predictor.sessions == {}, "nothing was loaded before the request was refused"
    ok("a verified frame outside the run's range is refused, not silently dropped")


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

    def refuses(label, pins, **kwargs):
        options = {"anchor": 0, "width": WIDTH, "height": HEIGHT}
        options.update(kwargs)
        try:
            _decode_pins(pins, **options)
        except Exception as exc:  # InvalidRequest / PropagateError
            assert "frame" in str(exc).lower(), str(exc)
            ok(label)
            return
        raise AssertionError(f"{label}: expected a rejection")

    refuses(
        "a pin on the anchor frame",
        [PinnedMask(frame_index=0, mask={"size": size, "counts": counts})],
    )
    refuses("a duplicated pin", [pin, pin])
    refuses(
        "an empty pin (that is a cleared frame, not a pin)",
        [
            PinnedMask(
                frame_index=9,
                mask={
                    "size": size,
                    "counts": encode_rle(np.zeros((HEIGHT, WIDTH), dtype=bool))[1],
                },
            )
        ],
    )
    refuses("a mask of the wrong size", [pin], width=WIDTH + 4)


def main():
    print("windowed propagation")
    try:
        test_windows_are_used_and_everything_is_covered()
        test_stitching_prefers_the_most_interior_window()
        test_the_anchor_frame_is_never_overwritten()
        test_later_windows_are_conditioned_on_the_overlap()
        test_tracking_is_forced_and_follows_the_window_direction()
        test_progress_callbacks_report_frames_and_windows()
        test_cancelling_stops_the_run_and_keeps_what_was_produced()
        test_backward_runs_are_planned_the_other_way()
        test_anchor_mask_shape()
        test_predictor_failures_are_not_swallowed()
        test_missing_frames_are_reported()
        test_propagation_runner_resolves_the_session_and_plans()
        test_verified_frames_anchor_and_are_never_overwritten()
        test_strict_chaining_stops_where_verification_ends()
        test_an_unknown_chaining_mode_is_rejected()
        test_the_runner_counts_verified_frames_out_of_progress()
        test_a_verified_frame_the_run_never_visits_is_rejected()
        test_pin_validation_rejects_unusable_frames()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
    print(f"\n{checks} checks passed")


if __name__ == "__main__":
    main()

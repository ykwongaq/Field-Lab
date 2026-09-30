"""Checks for the propagation window planner and its merging helpers.

Run from the backend root::

    python test/test_windows.py

Nothing here needs a GPU or a model: the planner is pure arithmetic, which is the
point of keeping it in `domain/`.
"""

import os
import sys

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from src.core.errors import InvalidRequest  # noqa: E402
from src.domain.windows import (  # noqa: E402
    DIRECTION_BACKWARD,
    DIRECTION_BOTH,
    DIRECTION_FORWARD,
    Window,
    covered_frames,
    interior_score,
    plan_full_clip,
    plan_windows,
    resolve_anchors,
)

checks = 0


def ok(label):
    global checks
    checks += 1
    print("  ok:", label)


def fails(label, call, expected=InvalidRequest):
    """Assert that `call` raises `expected`."""
    try:
        call()
    except expected:
        ok(label)
        return
    raise AssertionError(f"{label}: expected {expected.__name__}")


def test_forward_chain_covers_and_overlaps():
    plan = plan_windows(
        anchor=20,
        first=20,
        last=100,
        window_frames=16,
        overlap=4,
        direction=DIRECTION_FORWARD,
    )
    assert covered_frames(plan.windows) == set(range(20, 101)), "coverage"
    assert plan.windows[0].start == 20, "the first window starts at the anchor"
    for previous, current in zip(plan.windows, plan.windows[1:]):
        shared = set(previous.frames) & set(current.frames)
        assert len(shared) == 4, f"expected 4 shared frames, got {len(shared)}"
        assert current.start == previous.end - 4, "stride is window - overlap"
    assert all(window.direction == DIRECTION_FORWARD for window in plan.windows)
    ok(
        f"forward chain: {plan.windows_total} windows cover 20..100 with 4-frame overlaps"
    )


def test_backward_chain_ends_at_the_anchor():
    plan = plan_windows(
        anchor=20,
        first=0,
        last=20,
        window_frames=16,
        overlap=4,
        direction=DIRECTION_BACKWARD,
    )
    assert covered_frames(plan.windows) == set(range(0, 21)), "coverage"
    assert plan.windows[0].end == 21, "the first window ends just past the anchor"
    assert plan.windows[0].contains(20), "the anchor is inside the first window"
    assert all(window.direction == DIRECTION_BACKWARD for window in plan.windows)
    ok(f"backward chain: {plan.windows_total} windows cover 0..20")


def test_both_directions_go_forward_first():
    plan = plan_windows(
        anchor=50,
        first=0,
        last=100,
        window_frames=20,
        overlap=5,
        direction=DIRECTION_BOTH,
    )
    assert covered_frames(plan.windows) == set(range(0, 101)), "coverage"
    directions = [window.direction for window in plan.windows]
    assert directions == sorted(
        directions, key=lambda name: name != DIRECTION_FORWARD
    ), f"forward windows must come first, got {directions}"
    assert any(window.direction == DIRECTION_BACKWARD for window in plan.windows)
    ok(f"both directions: {plan.windows_total} windows, forward chain first")


def test_later_windows_carry_the_overlap_as_anchor_candidates():
    plan = plan_windows(
        anchor=0,
        first=0,
        last=40,
        window_frames=10,
        overlap=4,
        direction=DIRECTION_FORWARD,
    )
    first, second = plan.windows[0], plan.windows[1]
    assert first.overlap_frames == (
        0,
    ), "the first window is anchored on the caller's frame"
    assert second.overlap_frames == (6, 7, 8, 9), second.overlap_frames
    assert set(second.overlap_frames) <= set(
        first.frames
    ), "anchors come from the overlap"
    ok("window 2 is anchored on the overlap it shares with window 1")


def test_single_window_when_everything_fits():
    plan = plan_windows(
        anchor=3,
        first=0,
        last=7,
        window_frames=32,
        overlap=8,
        direction=DIRECTION_BOTH,
    )
    assert plan.windows_total == 1, plan.windows_total
    assert covered_frames(plan.windows) == set(range(0, 8))
    assert plan.frames_total == 8
    # One session walks both ways from the anchor instead of handing over between
    # a forward and a backward chain, which would cost a second session.
    assert plan.windows[0].direction == DIRECTION_BOTH
    assert plan.windows[0].overlap_frames == (3,), "anchored on the caller's frame"
    ok("a range shorter than the window is one session, walking both ways")


def test_full_clip_plan_defaults_to_everything():
    plan = plan_full_clip(anchor=100, frame_count=250, window_frames=48, overlap=8)
    assert covered_frames(plan.windows) == set(range(0, 250)), "coverage"
    assert plan.first == 0 and plan.last == 249
    payload = plan.payload()
    assert payload["windows_total"] == plan.windows_total
    assert payload["frames_total"] == 250
    assert len(payload["windows"]) == plan.windows_total
    ok(f"full clip: {plan.windows_total} windows cover the whole 250-frame clip")


def test_progress_counts_only_what_the_direction_produces():
    """A forward run must not count the frames before the anchor as outstanding."""
    forward = plan_windows(
        anchor=60,
        first=0,
        last=100,
        window_frames=16,
        overlap=4,
        direction=DIRECTION_FORWARD,
    )
    assert forward.payload()["frames_to_produce"] == 40, forward.payload()

    backward = plan_windows(
        anchor=60,
        first=0,
        last=100,
        window_frames=16,
        overlap=4,
        direction=DIRECTION_BACKWARD,
    )
    assert backward.payload()["frames_to_produce"] == 60, backward.payload()

    both = plan_windows(
        anchor=60,
        first=0,
        last=100,
        window_frames=16,
        overlap=4,
        direction=DIRECTION_BOTH,
    )
    assert both.payload()["frames_to_produce"] == 100, both.payload()
    # Overlapping windows must not double count coverage.
    assert both.payload()["frames_to_produce"] == both.covered - 1
    assert both.payload()["frames_total"] == 101, "the range itself is unchanged"
    ok("progress is counted against the frames the direction produces")


def test_interior_score_prefers_the_middle():
    window = Window(start=10, end=20, overlap_frames=(), direction=DIRECTION_FORWARD)
    assert interior_score(10, window) == 0
    assert interior_score(19, window) == 0
    assert interior_score(14, window) == 4
    assert interior_score(12, window) < interior_score(14, window)
    ok("interior score is highest in the middle of a window")


def test_resolve_anchors_filters_and_spreads():
    window = Window(
        start=20, end=30, overlap_frames=(16, 17, 18, 19), direction=DIRECTION_FORWARD
    )
    import numpy as np

    empty = np.zeros((4, 4), dtype=bool)
    full = np.ones((4, 4), dtype=bool)
    produced = {16: empty, 17: None, 18: full, 19: full}

    assert resolve_anchors(window, produced, 3) == (18, 19), "empty/None are skipped"
    assert resolve_anchors(window, produced, 1) == (
        19,
    ), "the last one wins when capped"
    assert resolve_anchors(window, {}, 3) == (), "nothing usable means no anchors"
    assert (
        resolve_anchors(window, produced, 0) == ()
    ), "max_anchors=0 disables anchoring"
    assert (
        len(resolve_anchors(window, {f: full for f in window.overlap_frames}, 3)) == 3
    )
    ok("anchor selection skips unusable frames, spreads, and respects the cap")


def test_validation_rejects_impossible_plans():
    fails(
        "unknown direction",
        lambda: plan_windows(
            anchor=1, first=0, last=9, window_frames=8, overlap=2, direction="sideways"
        ),
    )
    fails(
        "overlap as large as the window",
        lambda: plan_windows(
            anchor=1, first=0, last=9, window_frames=8, overlap=8, direction="both"
        ),
    )
    fails(
        "anchor outside the range",
        lambda: plan_windows(
            anchor=50, first=0, last=9, window_frames=8, overlap=2, direction="both"
        ),
    )
    fails(
        "last past the end of the clip",
        lambda: plan_windows(
            anchor=1,
            first=0,
            last=9,
            window_frames=8,
            overlap=2,
            direction="both",
            frame_count=5,
        ),
    )
    fails(
        "first after last",
        lambda: plan_windows(
            anchor=5, first=9, last=0, window_frames=8, overlap=2, direction="both"
        ),
    )
    fails(
        "negative anchor",
        lambda: plan_windows(
            anchor=-1, first=0, last=9, window_frames=8, overlap=2, direction="both"
        ),
    )
    ok("a plan that could not run is rejected up front")


def test_verified_frames_win_over_derived_ones():
    """A window holding a hand-verified mask is anchored on that alone.

    The other overlap frames carry the previous window's own prediction. Writing
    one of those into tracker memory as authoritative is what keeps a mistake
    alive across a boundary, so a verified frame displaces all of them.
    """
    import numpy as np

    window = Window(
        start=6, end=16, overlap_frames=(6, 7, 8, 9), direction=DIRECTION_FORWARD
    )
    full = np.ones((4, 4), dtype=bool)
    produced = {6: full, 7: full, 8: full, 9: full}

    assert resolve_anchors(window, produced, 3) == (6, 8, 9), "derived by default"
    assert resolve_anchors(window, produced, 3, pinned=(8,)) == (8,), "verified wins"
    assert resolve_anchors(window, produced, 3, pinned=(8,), allow_derived=False) == (
        8,
    )
    assert resolve_anchors(window, produced, 3, pinned=(6, 8, 9)) == (6, 8, 9)
    assert resolve_anchors(window, produced, 1, pinned=(6, 8, 9)) == (
        6,
    ), "the caller's anchor is never dropped by the cap"
    # A verified frame outside this window cannot condition it, and strict mode
    # then has nothing trustworthy to start the window from.
    assert resolve_anchors(window, produced, 3, pinned=(20,)) == (6, 8, 9)
    assert resolve_anchors(window, produced, 3, pinned=(20,), allow_derived=False) == ()
    assert (
        resolve_anchors(window, {6: None}, 3, pinned=(6,)) == ()
    ), "a frame with no mask cannot anchor"
    ok("a verified frame anchors a window, and its derived neighbours do not")


def test_a_chain_never_plans_a_one_frame_session():
    """A backward chain over just the anchor used to plan a 1-frame session.

    That session can only recompute a mask the caller already owns, so it costs a
    model load for nothing. The frames it covered are covered by the other chain.
    """
    plan = plan_windows(
        anchor=5,
        first=5,
        last=40,
        window_frames=10,
        overlap=4,
        direction=DIRECTION_BOTH,
    )
    assert all(window.length >= 2 for window in plan.windows), [
        (window.start, window.end) for window in plan.windows
    ]
    assert covered_frames(plan.windows) == set(range(5, 41)), "coverage is unchanged"
    assert plan.windows_total == 6, plan.windows_total
    ok("a chain never plans a one-frame session that only recomputes the anchor")


def test_payload_counts_the_verified_frames_out():
    """Pins are input, so they must not leave the progress bar short."""
    plan = plan_windows(
        anchor=20,
        first=20,
        last=100,
        window_frames=16,
        overlap=4,
        direction=DIRECTION_FORWARD,
    )
    assert plan.payload()["frames_to_produce"] == plan.covered - 1
    assert plan.payload(pinned=3)["frames_to_produce"] == plan.covered - 4
    assert plan.payload(pinned=3)["pinned"] == 3
    assert plan.payload(pinned=99)["frames_to_produce"] == 0, "never negative"
    assert plan.covers(100) and not plan.covers(
        19
    ), "only the frames the direction visits count as covered"
    ok("progress excludes the anchor and every hand-verified frame")


def main():
    print("window planner")
    test_forward_chain_covers_and_overlaps()
    test_backward_chain_ends_at_the_anchor()
    test_both_directions_go_forward_first()
    test_later_windows_carry_the_overlap_as_anchor_candidates()
    test_single_window_when_everything_fits()
    test_full_clip_plan_defaults_to_everything()
    test_progress_counts_only_what_the_direction_produces()
    test_interior_score_prefers_the_middle()
    test_resolve_anchors_filters_and_spreads()
    test_validation_rejects_impossible_plans()
    test_verified_frames_win_over_derived_ones()
    test_a_chain_never_plans_a_one_frame_session()
    test_payload_counts_the_verified_frames_out()
    print(f"\n{checks} checks passed")


if __name__ == "__main__":
    main()

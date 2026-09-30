"""Propagation window planning.

A clip is never propagated in one pass. The requested frame range is split into
a chain of *overlapping* windows, each of which becomes its own SAM 3 session:
peak GPU memory then depends on the window size rather than on the clip length,
which is what removes the old hard cap on how far a mask could be propagated.

Two properties make the chain work:

* **Coverage.** Consecutive windows overlap, so the frame range is covered with
  no holes and the hand-off happens on frames both windows predicted.
* **Re-anchoring.** The frames in the overlap carry masks the next window can be
  conditioned on, so the tracker's memory is warm when it takes over instead of
  starting cold at every boundary.

Stitching rule: a frame that falls inside two windows keeps the value from the
window where it is *most interior* (furthest from that window's edges), because
the frames at a window's edge are the ones most likely to be affected by the
window's truncation. This makes the merge deterministic instead of
last-writer-wins.

Everything here is pure: no torch, no numpy, no session access, so the planning
can be unit-tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidRequest

#: Propagation directions.
DIRECTION_FORWARD = "forward"
DIRECTION_BACKWARD = "backward"
DIRECTION_BOTH = "both"
DIRECTIONS = (DIRECTION_FORWARD, DIRECTION_BACKWARD, DIRECTION_BOTH)

#: How a window that holds no verified frame of its own may be seeded.
#:
#: `derived` hands it the previous window's own prediction. That is what lets a
#: run reach past a single window, and it is a documented compromise: the mask is
#: written into tracker memory as authoritative even though no human checked it.
#: `verified` refuses, so the chain stops instead and a run never travels further
#: than one window past a frame a human verified.
CHAINING_DERIVED = "derived"
CHAINING_VERIFIED = "verified"
CHAININGS = (CHAINING_DERIVED, CHAINING_VERIFIED)

#: The tracker attends at most four conditioning frames, so never ask for more.
MAX_ANCHORS = 4


@dataclass(frozen=True)
class Window:
    """One SAM 3 session's frame range, as a half-open `[start, end)` slice.

    ``overlap_frames`` are the frames whose masks this window can be conditioned
    on: for the first window of a chain that is the caller's anchor frame, and for
    every later window it is the stretch shared with the previous one. They are
    *candidates* — the planner cannot know which of them produced a usable mask,
    so the propagator filters them at run time (see `resolve_anchors`).
    """

    start: int
    end: int
    overlap_frames: Tuple[int, ...]
    direction: str

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def frames(self) -> range:
        return range(self.start, self.end)

    def contains(self, frame: int) -> bool:
        return self.start <= frame < self.end

    def local(self, frame: int) -> int:
        """The frame's index inside this window (what SAM 3 is given)."""
        if not self.contains(frame):
            raise InvalidRequest(f"Frame {frame} is not part of this window.")
        return frame - self.start

    def global_frame(self, local: int) -> int:
        """Inverse of `local`."""
        return self.start + local


@dataclass(frozen=True)
class PropagationPlan:
    """The ordered windows that cover `[first, last]` from `anchor`."""

    anchor: int
    first: int
    last: int
    direction: str
    windows: Tuple[Window, ...]

    @property
    def frames_total(self) -> int:
        return self.last - self.first + 1

    @property
    def covered(self) -> int:
        """How many distinct frames the windows cover.

        Consecutive windows overlap, so this is *not* the sum of their lengths.
        """
        frames: set = set()
        for window in self.windows:
            frames.update(window.frames)
        return len(frames)

    @property
    def windows_total(self) -> int:
        return len(self.windows)

    def covers(self, frame: int) -> bool:
        """True when one of the windows contains `frame`.

        Not every frame of the *requested* range is visited: a forward run over a
        range that starts before the anchor never touches those earlier frames.
        """
        return any(window.contains(frame) for window in self.windows)

    def payload(self, pinned: int = 0) -> dict:
        """A compact description for the API (no masks, just the shape of the run).

        `frames_to_produce` counts what the run will actually produce: the frames
        the *direction* covers, minus the anchor, which already carries the
        caller's mask, and minus `pinned`, which carries a mask a human verified
        by hand. A forward run over a range that starts before the anchor never
        touches those earlier frames, so counting them would leave the progress
        bar short of 100 % forever.
        """
        return {
            "anchor": self.anchor,
            "first": self.first,
            "last": self.last,
            "direction": self.direction,
            "frames_total": self.frames_total,
            "frames_to_produce": max(0, self.covered - 1 - max(0, int(pinned))),
            "pinned": max(0, int(pinned)),
            "windows_total": self.windows_total,
            "window_frames": max((w.length for w in self.windows), default=0),
            "windows": [
                {
                    "start": w.start,
                    "end": w.end,
                    "anchors": list(w.overlap_frames),
                    "direction": w.direction,
                }
                for w in self.windows
            ],
        }


def interior_score(frame: int, window: Window) -> int:
    """How far `frame` sits from the nearest edge of `window`.

    Frames at a window's boundary are the ones truncation can affect most, so the
    higher score wins when two windows disagree.
    """
    return min(frame - window.start, window.end - 1 - frame)


def _spread(candidates: Sequence[int], count: int) -> Tuple[int, ...]:
    """Pick `count` entries from `candidates`, keeping the first and the last."""
    if count >= len(candidates):
        return tuple(candidates)
    if count <= 1:
        return (candidates[-1],)
    last = len(candidates) - 1
    picked = {candidates[round(last * step / (count - 1))] for step in range(count)}
    return tuple(sorted(picked))


def resolve_anchors(
    window: Window,
    produced: Mapping[int, object],
    max_anchors: int = 3,
    *,
    pinned: Sequence[int] = (),
    allow_derived: bool = True,
) -> Tuple[int, ...]:
    """Choose which frames to condition this window on.

    `pinned` is the run's *verified* frame list in priority order — the caller's
    anchor first, then the frames a human corrected by hand. A mask from that list
    is the only thing worth writing into tracker memory as authoritative, so when
    a window contains one it is anchored on those and on nothing else: a derived
    hand-off must never carry a prediction across a frame a human verified.

    With no verified frame inside the window, the fallback is
    `window.overlap_frames`, which carries the previous window's own output. That
    is the documented compromise — it is what lets a run reach past one window —
    and `allow_derived=False` turns it off, in which case a window with no
    verified frame returns no anchors and the caller stops the chain there.

    Only frames that already hold a non-empty mask can anchor: conditioned on an
    empty mask the tracker would have nothing to track.
    """
    if max_anchors <= 0:
        return ()
    budget = min(max_anchors, MAX_ANCHORS)

    trusted = [
        frame
        for frame in pinned
        if window.contains(frame) and _has_mask(produced, frame)
    ]
    if trusted:
        # The first entry is the caller's anchor, which is never optional; the
        # rest are spread across the window so the tracker sees the object's
        # appearance at two ends of the stretch rather than twice in one place.
        keep = [trusted[0]]
        if budget > 1 and len(trusted) > 1:
            keep.extend(_spread(trusted[1:], budget - 1))
        return tuple(sorted(keep))

    if not allow_derived:
        return ()

    derived = [frame for frame in window.overlap_frames if _has_mask(produced, frame)]
    return _spread(derived, min(budget, len(derived)))


def _has_mask(produced: Mapping[int, object], frame: int) -> bool:
    """True when `frame` already carries a usable mask in `produced`.

    Empty and `None` entries are skipped: an empty mask, whether it came from a
    correction or from the tracker reporting nothing, cannot condition a window.
    """
    mask = produced.get(frame)
    return mask is not None and bool(mask.any())  # type: ignore[attr-defined]


def _forward_windows(anchor: int, last: int, size: int, overlap: int) -> List[Window]:
    """Windows that walk forward from `anchor`.

    The first window *starts* at the anchor so the conditioning frame is its
    first frame and the whole window is available for the propagation to cover.
    """
    windows: List[Window] = []
    start = anchor
    while start <= last:
        end = min(start + size, last + 1)
        if end <= start:
            break
        overlap_frames = (
            (anchor,) if not windows else tuple(range(start, start + overlap))
        )
        windows.append(Window(start, end, overlap_frames, DIRECTION_FORWARD))
        if end > last:
            break
        start = end - overlap
    return windows


def _backward_windows(anchor: int, first: int, size: int, overlap: int) -> List[Window]:
    """Windows that walk backward from `anchor`.

    Mirrored: the first window *ends* at the anchor (exclusive end), so the
    conditioning frame is its last frame.
    """
    windows: List[Window] = []
    end = anchor + 1
    while end > first:
        start = max(end - size, first)
        if start >= end:
            break
        overlap_frames = (anchor,) if not windows else tuple(range(end - overlap, end))
        windows.append(Window(start, end, overlap_frames, DIRECTION_BACKWARD))
        if start <= first:
            break
        end = start + overlap
    return windows


def plan_windows(
    *,
    anchor: int,
    first: int,
    last: int,
    window_frames: int,
    overlap: int,
    direction: str = DIRECTION_BOTH,
    frame_count: Optional[int] = None,
) -> PropagationPlan:
    """Split `[first, last]` into overlapping windows, starting at `anchor`.

    `first`/`last` are inclusive clip frame indices; `window_frames` is the
    number of frames one SAM 3 session may hold at once and `overlap` is how many
    frames consecutive windows share (the re-anchoring stretch).

    For `both`, the forward chain is planned first and the backward chain after
    it, matching the order SAM 3 itself propagates in, so the frames the caller
    is most likely looking at fill in first.
    """
    if direction not in DIRECTIONS:
        raise InvalidRequest(
            f"`direction` must be one of {', '.join(DIRECTIONS)}, got {direction!r}."
        )
    if window_frames < 1:
        raise InvalidRequest(
            f"`window_frames` must be at least 1, got {window_frames}."
        )
    if overlap < 0 or overlap >= window_frames:
        raise InvalidRequest(
            f"`overlap` must be at least 0 and smaller than `window_frames` "
            f"({window_frames}), got {overlap}."
        )
    if anchor < 0:
        raise InvalidRequest(f"`anchor` cannot be negative, got {anchor}.")
    if first > last:
        raise InvalidRequest(
            f"`first` ({first}) must not be greater than `last` ({last})."
        )
    if not first <= anchor <= last:
        raise InvalidRequest(
            f"`anchor` ({anchor}) must be inside the range [{first}, {last}]."
        )
    if frame_count is not None and last >= frame_count:
        raise InvalidRequest(
            f"`last` ({last}) is past the end of the clip ({frame_count} frames)."
        )

    size = window_frames
    span = last - first + 1
    if span <= size:
        # The whole range fits in one session, so plan exactly one. SAM 3 can walk
        # both ways from the anchor inside a single session, which is cheaper and
        # keeps one memory bank instead of handing over between two chains.
        return PropagationPlan(
            anchor=anchor,
            first=first,
            last=last,
            direction=direction,
            windows=(Window(first, last + 1, (anchor,), direction),),
        )

    windows: List[Window] = []
    if direction in (DIRECTION_FORWARD, DIRECTION_BOTH):
        windows.extend(_forward_windows(anchor, last, size, overlap))
    if direction in (DIRECTION_BACKWARD, DIRECTION_BOTH):
        windows.extend(_backward_windows(anchor, first, size, overlap))

    # A backward chain whose range is just the anchor plans a one-frame window
    # whose only frame is the anchor itself: a whole session to recompute a mask
    # the caller already owns. Drop it, and any other window too thin to be worth
    # a session (the frames it would cover are covered by the other chain).
    windows = [window for window in windows if window.length >= 2]

    return PropagationPlan(
        anchor=anchor,
        first=first,
        last=last,
        direction=direction,
        windows=tuple(windows),
    )


def plan_full_clip(
    *,
    anchor: int,
    frame_count: int,
    window_frames: int,
    overlap: int,
    direction: str = DIRECTION_BOTH,
) -> PropagationPlan:
    """Plan a run that covers the whole clip (`first = 0`, `last = frame_count - 1`).

    This is the default the reviewer offers: propagate both ways for as long as
    the clip lasts.
    """
    if frame_count < 1:
        raise InvalidRequest("The clip has no frames to propagate over.")
    return plan_windows(
        anchor=anchor,
        first=0,
        last=frame_count - 1,
        window_frames=window_frames,
        overlap=overlap,
        direction=direction,
        frame_count=frame_count,
    )


def covered_frames(windows: Iterable[Window]) -> set:
    """Union of the frames the windows cover (used by the planner's tests)."""
    covered: set = set()
    for window in windows:
        covered.update(window.frames)
    return covered

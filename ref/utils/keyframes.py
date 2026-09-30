"""Pick the keyframes of a track inside one sliding window.

The keyframes are the frames that will carry a *mask anchor*. They must be
frames where the track actually has a detection -- the dataset only stores
boxes on detected frames, and an anchor derived from a box on a frame where the
animal is absent would be meaningless.

For ``count=3`` this yields the first detected frame, the last detected frame,
and (subject to the farthest-point rule below) the detected frame closest to
the window centre.
"""

from __future__ import annotations

from typing import Iterable, Sequence

POSITION_PRESETS = {
    "first_mid_last": 3,
    "first_last": 2,
    "mid": 1,
}


def position_count(name: str) -> int:
    """Map a ``--keyframe_positions`` preset name to a keyframe count."""
    try:
        return POSITION_PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown keyframe preset {name!r}; expected one of "
            f"{sorted(POSITION_PRESETS)}"
        ) from None


def unique_sorted(frames: Iterable[int]) -> list[int]:
    """Sorted, de-duplicated integer frame indices."""
    return sorted({int(f) for f in frames})


def select_keyframes(
    detection_frames: Iterable[int],
    w_start: int,
    w_end: int,
    count: int = 3,
) -> list[int]:
    """Choose up to ``count`` keyframes for a track inside ``[w_start, w_end)``.

    Selection is farthest-point sampling seeded with the two extremes, so the
    result maximises the minimum spacing between anchors (which is exactly what
    bounds tracker drift). Ties are broken towards the window centre and then
    towards the lower frame index, so the result is deterministic.

    ``detection_frames`` may contain frames outside the window and duplicates;
    both are handled. Returns an ascending list, possibly shorter than ``count``
    when the track simply has fewer detections in this window.
    """
    count = int(count)
    if count <= 0:
        return []

    candidates = [f for f in unique_sorted(detection_frames) if w_start <= f < w_end]
    if not candidates:
        return []
    if len(candidates) <= count:
        return candidates

    centre = (w_start + w_end - 1) / 2.0
    if count == 1:
        return [min(candidates, key=lambda f: (abs(f - centre), f))]

    # Seed with the extremes, which are the frames that maximise coverage.
    chosen = [candidates[0], candidates[-1]]
    while len(chosen) < count:
        remaining = [f for f in candidates if f not in chosen]
        # Farthest-point insertion: maximise the distance to the nearest
        # already-chosen frame, tie-break towards the centre (then lower index).
        best = max(
            remaining,
            key=lambda f: (
                min(abs(f - c) for c in chosen),
                -abs(f - centre),
                -f,
            ),
        )
        chosen.append(best)

    return sorted(chosen)

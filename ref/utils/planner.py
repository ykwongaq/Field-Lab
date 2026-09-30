"""Keyframe anchor planning: keyframes -> probe -> accepted anchors (cached).

This is the orchestration layer between :mod:`utils.keyframes` (which frames to
use), :mod:`utils.probe` (what SAM3 predicts there) and :mod:`utils.anchors`
(whether the prediction is trustworthy enough to inject).

Results are cached per ``(track_id, frame_index)`` because the same detection
frame is a keyframe of more than one sliding window. A cache hit skips the
probe entirely, which is what makes re-running an experiment cheap.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .anchors import Anchor, AnchorCandidate, AnchorConfig, select_anchors
from .cache import append_jsonl, load_anchor_cache, save_anchor_cache
from .keyframes import select_keyframes
from .probe import ProbeResult

#: A track box: ``(frame_index, normalised_xywh, pixel_xyxy)``.
TrackBox = Tuple[int, Sequence[float], Sequence[float]]

#: A cache entry is only trusted when it carries all of these.
_REQUIRED_PROBE_KEYS = ("frame_index", "ok", "reason")


def _is_probe_payload(payload: Any) -> bool:
    """True when ``payload`` looks like a complete cached probe result.

    ``ProbeResult.from_cache_dict`` is deliberately lenient, so a partial write
    (or a hand-edited cache) would otherwise be accepted as a real result and
    never re-probed. Anything that fails this check is simply probed again.
    """
    if not isinstance(payload, dict):
        return False
    if not all(key in payload for key in _REQUIRED_PROBE_KEYS):
        return False
    if not isinstance(payload.get("ok"), bool):
        return False
    if payload["ok"] and payload.get("mask") is None:
        return False  # an accepted probe must carry its mask
    return True


class KeyframeAnchorPlanner:
    """Probe the keyframes of a track and return the anchors worth injecting.

    Args:
        probe: an :class:`~utils.probe.ImageModelProbe` /
            :class:`~utils.probe.VideoModelProbe`.
        config: the accept/reject policy.
        width, height: video resolution in pixels.
        frame_loader: ``frame_index -> PIL image`` for the window being tracked.
        keyframe_count: how many keyframes per track per window.
        cache_path: optional cache file; ``None`` disables caching.
        decisions_path: optional JSONL log of every accept/reject decision.
        log_lock: multiprocessing lock guarding the JSONL append.
        video_label: identifier written into the log/cache.
    """

    def __init__(
        self,
        probe,
        config: AnchorConfig,
        width: int,
        height: int,
        frame_loader: Callable[[int], Any],
        keyframe_count: int = 3,
        cache_path: Optional[str] = None,
        decisions_path: Optional[str] = None,
        log_lock=None,
        video_label: str = "",
    ) -> None:
        self.probe = probe
        self.config = config
        self.width = int(width)
        self.height = int(height)
        self.frame_loader = frame_loader
        self.keyframe_count = int(keyframe_count)
        self.cache_path = cache_path
        self.decisions_path = decisions_path
        self.log_lock = log_lock
        self.video_label = video_label

        self._entries: Dict[str, ProbeResult] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._pending_log: List[Dict[str, Any]] = []
        self._loaded = False
        self.stats: Dict[str, int] = {
            "tracks": 0,
            "keyframes": 0,
            "probes": 0,
            "cache_hits": 0,
            "accepted": 0,
            "rejected": 0,
            "errors": 0,
        }

    # -- cache ------------------------------------------------------------

    @staticmethod
    def _entry_key(track_id: int, frame_index: int) -> str:
        return f"{int(track_id)}:{int(frame_index)}"

    def _load_cache(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.cache_path:
            return
        payload = load_anchor_cache(self.cache_path) or {}
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            probe_payload = entry.get("probe")
            if not _is_probe_payload(probe_payload):
                continue  # corrupt/partial entry -> probe it again
            try:
                self._entries[str(key)] = ProbeResult.from_cache_dict(probe_payload)
            except (KeyError, TypeError, ValueError):
                continue
            self._meta[str(key)] = {
                "track_id": entry.get("track_id"),
                "species": entry.get("species"),
                "frame_index": entry.get("frame_index"),
            }

    def flush(self) -> None:
        """Persist the probe cache and append the pending decision records."""
        if self.cache_path:
            payload = {
                "video": self.video_label,
                "entries": {
                    key: {
                        **self._meta.get(key, {}),
                        "probe": probe.to_cache_dict(),
                    }
                    for key, probe in self._entries.items()
                },
            }
            save_anchor_cache(self.cache_path, payload)
        if self.decisions_path and self._pending_log:
            append_jsonl(self.decisions_path, self._pending_log, lock=self.log_lock)
        self._pending_log = []

    # -- planning ---------------------------------------------------------

    def _probe_keyframe(
        self, species: str, frame: int, box: Sequence[float]
    ) -> ProbeResult:
        """Probe one keyframe, failing *open* (never raising into the caller)."""
        try:
            image = self.frame_loader(frame)
            results = self.probe.probe_image(
                image,
                [(species, tuple(float(v) for v in box))],
                frame_index=frame,
            )
        except Exception as exc:  # a probe must never kill a video
            self.stats["errors"] += 1
            return ProbeResult(
                frame_index=frame,
                ok=False,
                reason=f"probe_error:{type(exc).__name__}",
            )
        self.stats["probes"] += 1
        if not results:
            return ProbeResult(frame_index=frame, ok=False, reason="no_result")
        return results[0]

    def anchors_for_track(
        self,
        track_id: int,
        species: str,
        boxes: Sequence[TrackBox],
        w_start: int,
        w_end: int,
    ) -> List[Anchor]:
        """Return the accepted mask anchors for one track inside one window."""
        self._load_cache()
        track_id = int(track_id)

        pixels: Dict[int, Sequence[float]] = {}
        for frame_index, _norm, pixel_box in boxes:
            pixels[int(frame_index)] = pixel_box

        keyframes = select_keyframes(pixels.keys(), w_start, w_end, self.keyframe_count)
        if not keyframes:
            return []

        self.stats["tracks"] += 1
        self.stats["keyframes"] += len(keyframes)

        for frame in keyframes:
            entry_key = self._entry_key(track_id, frame)
            if entry_key in self._entries:
                self.stats["cache_hits"] += 1
                continue
            self._entries[entry_key] = self._probe_keyframe(
                species, frame, pixels[frame]
            )
            self._meta[entry_key] = {
                "track_id": track_id,
                "species": species,
                "frame_index": frame,
            }

        candidates: List[AnchorCandidate] = []
        probe_reasons: Dict[int, str] = {}
        for frame in keyframes:
            probe = self._entries[self._entry_key(track_id, frame)]
            probe_reasons[frame] = probe.reason
            candidates.append(
                AnchorCandidate(
                    frame_index=frame,
                    mask=probe.mask if probe.ok else None,
                    score=probe.score if probe.ok else None,
                    gt_box=pixels[frame],
                )
            )

        anchors, decisions = select_anchors(
            candidates, self.config, self.width, self.height
        )

        for decision in decisions:
            self.stats["accepted" if decision.accepted else "rejected"] += 1
            self._pending_log.append(
                {
                    "video": self.video_label,
                    "track_id": track_id,
                    "species": species,
                    "window": [int(w_start), int(w_end)],
                    "frame_index": int(decision.frame_index),
                    "accepted": bool(decision.accepted),
                    "reason": decision.reason,
                    "probe_reason": probe_reasons.get(decision.frame_index),
                    "score": decision.score,
                    "containment": decision.containment,
                    "area": decision.area,
                }
            )

        return anchors

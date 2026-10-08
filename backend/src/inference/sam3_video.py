"""Propagating one mask through a clip, a window at a time.

The clip is never handed to SAM 3 all at once. `domain.windows` splits the
requested range into overlapping windows; each window becomes its own SAM 3
session, is conditioned on masks the previous window already produced, and is
closed before the next one opens. Peak memory is therefore a function of the
window size and not of the clip length, which is what makes "propagate to the
end" a reasonable thing for a reviewer to click.

The request sequence per window is the one the reference implementation uses
(`ref/utils/tracking.py`), minus the text/box prompt, because here the prompt is
the mask itself:

1. ``start_session``      - load the window's frames (in memory, no temp files)
2. ``add_mask`` (xN)      - write the conditioning frames into tracker memory
3. ``propagate_in_video`` - with ``force_tracker_propagation`` so the tracker
                            actually runs rather than replaying cached detector
                            predictions
4. ``close_session``

Every step is driven through `handle_request`/`handle_stream_request`, so the
whole thing is testable with a stub predictor and no GPU.

Each window holds `ModelManager.gate` for its own duration rather than the run
holding it from end to end, so an interactive prompt waits for one window and not
for a whole clip. See `inference.gate`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from src.core.config import (
    DEFAULT_PROPAGATE_ANCHOR_MAX,
    DEFAULT_PROPAGATE_CHAINING,
    DEFAULT_PROPAGATE_OVERLAP,
    DEFAULT_PROPAGATE_WINDOW_FRAMES,
    Settings,
)
from src.core.errors import InvalidRequest, Unavailable
from src.core.jobs import PropagationJob
from src.core.sessions import Session, open_session
from src.domain.windows import (
    CHAINING_DERIVED,
    CHAININGS,
    DIRECTION_BACKWARD,
    DIRECTION_BOTH,
    PropagationPlan,
    Window,
    interior_score,
    plan_windows,
    resolve_anchors,
)
from src.inference.frames import load_frames, uniform_size
from src.inference.models import ModelKind, ModelManager, Sam3Config

#: The single object each window tracks. Sessions are independent and the
#: reviewer owns object identity, so there is no need for a global id space.
OBJECT_ID = 1


class PropagateError(InvalidRequest):
    """The propagation request cannot be served."""


class PropagateUnavailable(Unavailable):
    """The tracker is not available (not installed, no CUDA, load failure)."""


#: `(frame_index, mask)` for every mask as it is produced.
MaskCallback = Callable[[int, np.ndarray], None]
#: `(window_index, windows_total)` whenever a new window starts.
WindowCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class PropagateConfig:
    """Window sizing for propagation, and how a window may be seeded."""

    #: Defaults come from `core.config`, which is where `config/server.json` and
    #: the environment are read, so each value has one source of truth instead of
    #: a second copy here that can drift out of step with it.
    window_frames: int = DEFAULT_PROPAGATE_WINDOW_FRAMES
    overlap: int = DEFAULT_PROPAGATE_OVERLAP
    anchor_max: int = DEFAULT_PROPAGATE_ANCHOR_MAX
    #: `derived` (default) lets a window with no verified frame be seeded from the
    #: previous window's own output; `verified` stops the chain there instead.
    chaining: str = DEFAULT_PROPAGATE_CHAINING

    def __post_init__(self) -> None:
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        if not 0 <= self.overlap < self.window_frames:
            raise ValueError("overlap must be smaller than window_frames")
        if self.anchor_max < 0:
            raise ValueError("anchor_max must be >= 0")
        if self.chaining not in CHAININGS:
            raise ValueError(
                f"chaining must be one of {', '.join(CHAININGS)}, "
                f"got {self.chaining!r}"
            )

    @property
    def allow_derived(self) -> bool:
        """Whether an unverified window may be seeded from a prediction."""
        return self.chaining == CHAINING_DERIVED

    @classmethod
    def from_settings(cls, settings: Settings) -> "PropagateConfig":
        return cls(
            window_frames=settings.propagate_window_frames,
            overlap=settings.propagate_overlap,
            anchor_max=settings.propagate_anchor_max,
            chaining=settings.propagate_chaining,
        )


class Sam3VideoPropagator:
    """Windowed mask propagation over one session's frames."""

    def __init__(
        self,
        config: Optional[PropagateConfig] = None,
        manager: Optional[ModelManager] = None,
        *,
        model_config: Optional[Sam3Config] = None,
        predictor_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config or PropagateConfig()
        self._manager = manager or ModelManager(model_config)
        self._predictor_provider = predictor_provider or self._manager.video

    # ── availability ────────────────────────────────────────────────────

    @staticmethod
    def installed() -> bool:
        return ModelManager.installed()

    def status(self) -> Dict[str, Any]:
        """Tracker availability, the window sizing in force, and gate occupancy."""
        status = self._manager.status()
        status["window_frames"] = self.config.window_frames
        status["overlap"] = self.config.overlap
        status["anchor_max"] = self.config.anchor_max
        status["chaining"] = self.config.chaining
        # Whether a model call is in flight, and how many callers are queued for
        # it. The gate is process-wide, so this covers prompts as well as runs:
        # it is the number that explains a click which is slow because a
        # propagation window currently holds the models.
        status["gpu_busy"] = self._manager.gate.busy
        status["gpu_waiting"] = self._manager.gate.waiting
        return status

    def warmup(self) -> None:
        """Load the predictor now."""
        self._manager.warmup(kind=ModelKind.VIDEO)

    # ── planning ────────────────────────────────────────────────────────

    def plan(
        self,
        *,
        anchor: int,
        first: int,
        last: int,
        direction: str,
        frame_count: int,
    ) -> PropagationPlan:
        """Plan the windows for a run without touching a model."""
        return plan_windows(
            anchor=anchor,
            first=first,
            last=last,
            direction=direction,
            window_frames=self.config.window_frames,
            overlap=self.config.overlap,
            frame_count=frame_count,
        )

    # ── propagation ─────────────────────────────────────────────────────

    def propagate(
        self,
        session: Session,
        *,
        anchor: int,
        anchor_mask: np.ndarray,
        plan: PropagationPlan,
        pins: Optional[Mapping[int, Any]] = None,
        chaining: Optional[str] = None,
        on_mask: Optional[MaskCallback] = None,
        on_window: Optional[WindowCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> Dict[int, np.ndarray]:
        """Run `plan`, returning `{frame_index: mask}` for every frame covered.

        `pins` are the frames a human verified by hand, other than the anchor.
        They are treated exactly like the anchor: seeded into the tracker as
        authoritative, and never overwritten by the run. Model output is never a
        pin — see `resolve_anchors` for why that matters.

        Masks are reported through `on_mask` as they are produced, so a caller can
        paint them immediately instead of waiting for the run to finish. The
        anchor frame is never overwritten: the caller's mask *is* the ground truth
        for that frame, and a propagation that disagrees with it is wrong.
        """
        if chaining is not None and chaining not in CHAININGS:
            raise PropagateError(
                f"`chaining` must be one of {', '.join(CHAININGS)}, got {chaining!r}."
            )
        allow_derived = (
            self.config.allow_derived
            if chaining is None
            else chaining == CHAINING_DERIVED
        )

        anchor_mask = _as_bool_mask(anchor_mask)
        verified = _pin_masks(pins, anchor)
        produced: Dict[int, np.ndarray] = {**verified, anchor: anchor_mask}
        # A verified frame outranks anything a window computes for it.
        best_score: Dict[int, float] = {frame: float("inf") for frame in produced}
        # Priority order: the caller's anchor first, then the verified frames by
        # index, so a window that has to choose keeps the mask the user drew.
        order: Tuple[int, ...] = (anchor, *sorted(f for f in verified if f != anchor))
        cancel = cancel or threading.Event()

        predictor = self._predictor_provider()
        total = plan.windows_total
        for index, window in enumerate(plan.windows, start=1):
            if cancel.is_set():
                break
            anchors = resolve_anchors(
                window,
                produced,
                self.config.anchor_max,
                pinned=order,
                allow_derived=allow_derived,
            )
            if not anchors and not allow_derived:
                # Strict chaining: this window holds no verified mask, so there is
                # nothing trustworthy to start it from. Stop rather than guess.
                break
            if on_window is not None:
                on_window(index, total)
            self._run_window(
                predictor,
                session,
                window,
                anchors,
                produced,
                best_score,
                on_mask,
                cancel,
            )
        return produced

    def _run_window(
        self,
        predictor: Any,
        session: Session,
        window: Window,
        anchors: Sequence[int],
        produced: Dict[int, np.ndarray],
        best_score: Dict[int, float],
        on_mask: Optional[MaskCallback],
        cancel: threading.Event,
    ) -> None:
        """One SAM 3 session: condition it on `anchors`, propagate, close it."""
        frames = load_frames(session, window.frames)
        uniform_size(frames)

        # Two contexts, both released together at the end of the window:
        #  * the gate, so only one model call runs at a time and a prompt waits
        #    for a window rather than for the whole run;
        #  * one autocast context covering the conditioning masks and the stream
        #    they condition; see `ModelManager.inference_context`.
        with self._manager.gate.batch(), self._manager.inference_context():
            session_id = self._start_session(predictor, window, frames)
            del frames  # the session keeps its own copy
            try:
                for frame in anchors:
                    predictor.handle_request(
                        request=dict(
                            type="add_mask",
                            session_id=session_id,
                            frame_index=window.local(frame),
                            obj_id=OBJECT_ID,
                            mask=_as_bool_mask(produced[frame]),
                        )
                    )
                self._propagate_window(
                    predictor,
                    session_id,
                    window,
                    _reverse_start(window, anchors),
                    produced,
                    best_score,
                    on_mask,
                    cancel,
                )
            finally:
                self._close_session(predictor, session_id)

    def _propagate_window(
        self,
        predictor: Any,
        session_id: str,
        window: Window,
        start_frame: Optional[int],
        produced: Dict[int, np.ndarray],
        best_score: Dict[int, float],
        on_mask: Optional[MaskCallback],
        cancel: threading.Event,
    ) -> None:
        """Consume the window's stream, keeping the most interior prediction."""
        request = dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction=window.direction,
            # `add_mask` records a refinement, which can make the action
            # history replay cached predictions instead of tracking.
            force_tracker_propagation=True,
        )
        if start_frame is not None:
            # Without this SAM 3 starts a reverse walk from the *earliest*
            # injected mask, which is wrong when a verified frame sits below the
            # frame anchoring the hand-off; see `_reverse_start`.
            request["start_frame_index"] = start_frame
        for response in predictor.handle_stream_request(request=request):
            if cancel.is_set():
                return
            local_frame = int(response["frame_index"])
            # The stream indexes frames *within the window*; `Window.contains`
            # speaks global indices, so the bound is the window's length.
            if not 0 <= local_frame < window.length:
                continue
            global_frame = window.global_frame(local_frame)
            current = best_score.get(global_frame)
            if current == float("inf"):
                continue  # the caller's anchor mask is authoritative
            mask = _best_mask(response.get("outputs"))
            if mask is None:
                continue
            score = float(interior_score(global_frame, window))
            if current is not None and score < current:
                continue
            best_score[global_frame] = score
            produced[global_frame] = mask
            if on_mask is not None:
                on_mask(global_frame, mask)

    # ── predictor plumbing ──────────────────────────────────────────────

    def _start_session(
        self, predictor: Any, window: Window, frames: Sequence[Any]
    ) -> str:
        """Open a session on the window's frames, already decoded and in memory."""
        try:
            response = predictor.handle_request(
                request=dict(type="start_session", resource_path=list(frames))
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                _free_cuda()
                raise PropagateUnavailable(
                    f"CUDA ran out of memory on a {window.length}-frame window; lower "
                    "`propagate.window_frames` in config/server.json and run again."
                ) from exc
            raise
        return str(response["session_id"])

    @staticmethod
    def _close_session(predictor: Any, session_id: str) -> None:
        try:
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
        except Exception:  # noqa: BLE001 - closing must never mask the real error
            pass


def _reverse_start(window: Window, anchors: Sequence[int]) -> Optional[int]:
    """The window-local frame a *reverse* walk must begin below, or `None`.

    SAM 3 decides where a propagation pass starts from the earliest frame that
    carries an injected mask (``previous_stages_out`` in the model's
    ``_get_processing_order``), not from the frame the caller is anchored on. That
    is fine while a window holds a single anchor, but it breaks a backward window
    that is seeded with a verified frame *below* its hand-off frame: the reverse
    walk then starts at the lower mask and walks down, so the frames between the
    pin and the anchor are never tracked at all. A re-run seeded from a corrected
    frame therefore punches a hole in the mask between that frame and the anchor.

    Asking for the highest anchor instead makes the walk descend through every
    frame of the window; the lower pin is still honoured, because the tracker
    reuses the injected output it stored for that frame when the walk reaches it.

    Only a backward window needs the override: a forward walk already starts at
    the earliest injected frame, which *is* the correct end to ascend from.
    """
    if window.direction != DIRECTION_BACKWARD or not anchors:
        return None
    return window.local(max(anchors))


def _pin_masks(pins: Optional[Mapping[int, Any]], anchor: int) -> Dict[int, np.ndarray]:
    """The caller's verified frames as bool masks, minus the anchor's own frame.

    The anchor travels separately in `anchor_mask` because it is not optional — a
    run without it has no object to track — so a pin repeating that frame is
    dropped rather than allowed to duplicate it.
    """
    out: Dict[int, np.ndarray] = {}
    for frame, mask in (pins or {}).items():
        index = int(frame)
        if index == anchor:
            continue
        out[index] = _as_bool_mask(mask)
    return out


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    """`mask` as a 2-D bool array (a no-op copy when it already is)."""
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise PropagateError(f"A mask must be 2-D, got shape {array.shape}.")
    return array.astype(bool, copy=False)


def _best_mask(outputs: Any) -> Optional[np.ndarray]:
    """The mask of the tracked object in one frame's outputs.

    A window tracks exactly one object, so at most one entry matters; the largest
    non-empty mask wins, which keeps the result stable if the tracker ever reports
    a spurious extra object.
    """
    if not outputs:
        return None
    masks = outputs.get("out_binary_masks")
    if masks is None:
        return None
    array = np.asarray(masks)
    if array.size == 0:
        return None
    if array.ndim == 2:
        array = array[None, ...]
    best: Optional[np.ndarray] = None
    best_area = 0
    for candidate in array:
        binary = np.asarray(candidate) > 0
        area = int(binary.sum())
        if area > best_area:
            best, best_area = binary, area
    return best


def _free_cuda() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class PropagationRunner:
    """Adapts a queued `PropagationJob` to `Sam3VideoPropagator.propagate`.

    Lives here rather than in the registry so that `core.jobs` stays ignorant of
    sessions and models, and the whole job lifecycle can be tested with a fake
    runner.
    """

    def __init__(self, propagator: Sam3VideoPropagator, settings: Settings) -> None:
        self.propagator = propagator
        self.settings = settings

    def __call__(self, job: PropagationJob) -> None:
        if job.anchor_mask is None:
            raise PropagateError("The job has no anchor mask to propagate.")
        session = open_session(
            self.settings.temp_dir, job.session_id, owner=job.client_id
        )
        session.touch()

        plan = self.propagator.plan(
            anchor=job.anchor,
            first=job.first,
            last=job.last,
            direction=job.direction,
            frame_count=job.frame_count or session.frame_count(),
        )
        # A verified frame the plan never visits could not be injected either —
        # SAM 3 addresses frames inside one session — so it is a mistake in the
        # request rather than a pin that quietly does nothing.
        outside = sorted(frame for frame in job.pins if not plan.covers(frame))
        if outside:
            raise PropagateError(
                f"Verified frame(s) {outside} are outside the frames this run "
                f"covers ({plan.first}..{plan.last}, {plan.direction})."
            )
        job.set_plan(plan.payload(pinned=len(job.pins)))

        started = time.perf_counter()
        masks = self.propagator.propagate(
            session,
            anchor=job.anchor,
            anchor_mask=job.anchor_mask,
            plan=plan,
            pins=job.pins,
            chaining=job.chaining,
            on_mask=job.add_mask,
            on_window=job.set_window,
            cancel=job.cancel,
        )
        job.check_cancelled()
        if job.plan is not None:
            job.plan["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            # `masks` carries the anchor and every verified frame as well; neither
            # is something the run produced.
            job.plan["frames_produced"] = max(0, len(masks) - 1 - len(job.pins))
        session.touch()

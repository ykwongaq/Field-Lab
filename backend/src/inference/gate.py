"""Admission control for the one accelerator.

There is exactly one GPU and two kinds of work that want it: a prompt (short,
interactive) and a propagation window (long, batch). Neither is re-entrant, and
the model budget is shared, so a second caller is not merely slower — it can
evict the weights the first one is still using. With
``sam3.max_resident_models = 1`` a click that loads the image model releases the
video predictor, and releasing the video predictor calls ``shutdown()`` on it,
which tears down the worker processes of a run that may be mid-stream.

So every model call goes through this gate. Two classes, because they want
different things:

* ``interactive`` — one prompt. Waits for whatever is running, and never behind
  a run that has *not* started, so a click cannot queue behind a propagation.
* ``batch`` — one propagation window. A run re-enters the gate for each window
  and releases it in between, so an interactive request waits at most one window
  (about ``propagate.window_frames`` frames) rather than the whole run.

The priority rule is that prompts go first, and it is a *bounded* priority: a run
lets `DEFAULT_BATCH_YIELDS` prompts go ahead of it and then takes its turn even
though prompts are still waiting. Without that counterweight a steady click stream
would starve propagation, and a run on a busy server could never reclaim the
accelerator.

Waiters are queued and a releasing holder *names* its successor. That detail is
load-bearing rather than incidental: waking every waiter to race for the lock lets
the same class win repeatedly, which is the starvation this module exists to
prevent, and it makes the bound above true only on average. It was written the
racing way first, and a test caught it — with a deferral budget of one, a
propagation was still served last behind six prompts.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Iterator, List, Optional

#: How many prompts a batch waiter lets go ahead of it before it takes its turn
#: anyway. Small enough that a run still makes progress under a continuous stream
#: of clicks, large enough that ordinary click-by-click refinement is never held
#: up by a propagation. This is scheduler policy rather than a deployment
#: setting, so it lives beside the policy; promote it to `Settings` if it ever
#: needs tuning per site.
DEFAULT_BATCH_YIELDS = 8


@dataclass(eq=False)
class _Waiter:
    """One caller queueing for the gate.

    `eq=False` on purpose: two waiters with identical fields are still two
    callers, and the dispatcher identifies the chosen one by identity when it
    hands the gate over.
    """

    interactive: bool
    #: For a batch waiter, how many more prompts may go ahead of it. Reaching
    #: zero is what stops a click stream from starving a run.
    yields: int = 0
    granted: bool = False


class ModelGate:
    """A one-at-a-time lock over the models, with a bounded interactive priority."""

    def __init__(self, *, batch_yields: int = DEFAULT_BATCH_YIELDS) -> None:
        self._batch_yields = max(0, batch_yields)
        self._condition = threading.Condition()
        self._busy = False
        #: Callers waiting for the gate, oldest first. The holder is not in here.
        self._waiters: List[_Waiter] = []

    # ── the two classes of caller ───────────────────────────────────────

    @contextlib.contextmanager
    def interactive(self) -> Iterator[None]:
        """Hold the models for one prompt (short; never queued behind a queue)."""
        self._enqueue(interactive=True)
        try:
            yield
        finally:
            self._release()

    @contextlib.contextmanager
    def batch(self) -> Iterator[None]:
        """Hold the models for one propagation window, then hand them back."""
        self._enqueue(interactive=False)
        try:
            yield
        finally:
            self._release()

    # ── observability ───────────────────────────────────────────────────

    @property
    def busy(self) -> bool:
        """Whether a model call is in flight right now."""
        with self._condition:
            return self._busy

    @property
    def waiting(self) -> int:
        """How many callers are queued for the gate right now, holder aside."""
        with self._condition:
            return len(self._waiters)

    # ── internals ───────────────────────────────────────────────────────

    def _enqueue(self, *, interactive: bool) -> None:
        """Join the queue and block until the gate is handed over."""
        waiter = _Waiter(
            interactive=interactive,
            yields=0 if interactive else self._batch_yields,
        )
        with self._condition:
            self._waiters.append(waiter)
            self._dispatch_locked()
            while not waiter.granted:
                # Re-checked in a loop because a wake-up only means "the gate
                # moved", not "this waiter is next".
                self._condition.wait()

    def _release(self) -> None:
        with self._condition:
            self._busy = False
            self._dispatch_locked()

    def _dispatch_locked(self) -> None:
        """Hand the gate to the next waiter, if the policy says there is one.

        Called with the condition held, from both ends: a waiter arriving at an
        idle gate takes it immediately, and a holder leaving hands it on.
        """
        if self._busy or not self._waiters:
            return
        chosen = self._next_locked()
        self._waiters.remove(chosen)
        if chosen.interactive:
            # A prompt is going ahead of every queued run, so charge each of them
            # a deferral. This is the whole accounting behind the guarantee that
            # a click stream cannot starve a run.
            for waiter in self._waiters:
                if not waiter.interactive:
                    waiter.yields -= 1
        self._busy = True
        chosen.granted = True
        self._condition.notify_all()

    def _next_locked(self) -> Optional[_Waiter]:
        """Who goes next: a run that has waited long enough, else a prompt."""
        waited_enough = [
            waiter
            for waiter in self._waiters
            if not waiter.interactive and waiter.yields <= 0
        ]
        if waited_enough:
            return waited_enough[0]
        prompts = [waiter for waiter in self._waiters if waiter.interactive]
        if prompts:
            # Oldest prompt first: prompts are first-come, first-served among
            # themselves, which is all a click can ask for.
            return prompts[0]
        return self._waiters[0]

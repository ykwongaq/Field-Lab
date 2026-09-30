"""Checks for the model gate: one model call at a time, prompts first.

No GPU and no model are involved: the gate is pure threading, which is the point
of keeping the admission policy out of the model wrapper. Two properties have to
hold, and they pull against each other:

* **Mutual exclusion** — a second caller waits for the holder rather than racing
  it, because loading or releasing a model under another caller's feet is what
  used to tear down a running predictor.
* **Fairness both ways** — a propagation must not block a reviewer's click for a
  whole clip, and a steady stream of clicks must not stop a propagation from ever
  reclaiming the accelerator.
"""

from __future__ import annotations

import threading
import time

import pytest

from src.inference.gate import ModelGate

#: Generous, because every wait below is a handshake: a real hang must fail, and
#: a slow machine must not.
TIMEOUT = 10.0


def wait_until(predicate, label, timeout=TIMEOUT):
    """Block until `predicate` holds, so no test depends on a fixed sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError(f"timed out waiting for {label}")


def hold(gate, kind):
    """The context manager a caller of `kind` would use."""
    return gate.interactive() if kind == "prompt" else gate.batch()


class OverlapDetector:
    """Counts concurrent holders, so an overlap is observed rather than inferred."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.005)  # hold long enough for a rival to overlap if it could
        with self._lock:
            self.active -= 1


def test_a_second_caller_waits_for_the_holder():
    """The gate is exclusive, and it is handed over rather than dropped."""
    gate = ModelGate()
    entered = threading.Event()

    def waiter():
        with gate.interactive():
            entered.set()

    with gate.interactive():
        thread = threading.Thread(target=waiter)
        thread.start()
        wait_until(lambda: gate.waiting == 1, "the waiter to queue")
        assert not entered.wait(0.2), "the waiter entered while the gate was held"

    assert entered.wait(TIMEOUT), "the waiter was never released"
    thread.join(TIMEOUT)
    assert not thread.is_alive(), "the waiter never finished"
    assert not gate.busy and gate.waiting == 0, "the gate leaked its state"


def test_a_prompt_overtakes_a_waiting_propagation():
    """A click that queued behind a run is served before the run resumes."""
    gate = ModelGate()
    order = []

    def take(kind):
        with hold(gate, kind):
            order.append(kind)

    with gate.batch():
        propagation = threading.Thread(target=take, args=("propagation",))
        propagation.start()
        wait_until(lambda: gate.waiting == 1, "the propagation to queue")
        prompt = threading.Thread(target=take, args=("prompt",))
        prompt.start()
        wait_until(lambda: gate.waiting == 2, "the prompt to queue")

    prompt.join(TIMEOUT)
    propagation.join(TIMEOUT)
    assert order == ["prompt", "propagation"], order
    assert not gate.busy and gate.waiting == 0, "the gate leaked its state"


def test_an_exception_still_releases_the_gate():
    """A failed model call must not wedge the accelerator for everyone else."""
    gate = ModelGate()
    with pytest.raises(RuntimeError):
        with gate.batch():
            raise RuntimeError("boom")

    assert not gate.busy, "the gate stayed busy after the body raised"
    assert gate.waiting == 0, "the gate leaked a waiter"
    with gate.interactive():
        pass


def test_concurrent_callers_never_overlap():
    """Prompts and propagations from several threads, one holder at a time."""
    gate = ModelGate()
    detector = OverlapDetector()
    failures = []

    def worker(kind, rounds):
        try:
            for _ in range(rounds):
                with hold(gate, kind):
                    detector.enter()
        except Exception as exc:  # noqa: BLE001 - reported, not raised in a thread
            failures.append(f"{kind}: {exc!r}")

    rounds = 12
    threads = [
        threading.Thread(target=worker, args=(kind, rounds))
        for kind in ("prompt", "propagation") * 3
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT * 3)

    assert not any(thread.is_alive() for thread in threads), "a worker deadlocked"
    assert not failures, failures
    assert detector.calls == rounds * len(threads), "the workload did not run"
    assert detector.peak == 1, f"two model calls overlapped (peak {detector.peak})"
    assert not gate.busy and gate.waiting == 0, "the gate leaked its state"


def queued_run(batch_yields, prompt_count):
    """Hold the gate, queue `prompt_count` prompts plus a propagation, release.

    Returns the order the waiters were served in, which is the only thing the
    deferral budget decides: how many prompts go ahead of a run before it takes
    its turn.
    """
    gate = ModelGate(batch_yields=batch_yields)
    order = []
    recorder = threading.Lock()

    def serve(kind):
        with hold(gate, kind):
            with recorder:
                order.append(kind)

    threads = []
    with gate.interactive():
        for _ in range(prompt_count):
            thread = threading.Thread(target=serve, args=("prompt",))
            thread.start()
            threads.append(thread)
        propagation = threading.Thread(target=serve, args=("propagation",))
        propagation.start()
        threads.append(propagation)
        wait_until(lambda: gate.waiting == prompt_count + 1, "every waiter to queue")

    for thread in threads:
        thread.join(TIMEOUT)
    assert not any(thread.is_alive() for thread in threads), "a waiter deadlocked"
    assert len(order) == prompt_count + 1, order
    return gate, order


def test_a_run_still_defers_while_it_has_budget():
    """With budget to spare, a run lets every queued prompt through first.

    This is the behaviour a reviewer wants: a click never waits for a clip.

    The exact order is asserted, not just "the run is not last": dispatch is
    deterministic, so anything looser would not notice a regression in *when* a
    waiter is served, only in whether it eventually is.
    """
    gate, order = queued_run(batch_yields=100, prompt_count=4)
    assert order == ["prompt"] * 4 + ["propagation"], order
    assert not gate.busy and gate.waiting == 0, "the gate leaked its state"


def test_a_run_takes_its_turn_once_its_deferrals_are_spent():
    """A budget of N lets exactly N prompts ahead, and then the run goes.

    Without this the rule "a run defers to any waiting prompt" has no
    counterweight, and on a busy server a propagation would never reclaim the
    accelerator. One prompt ahead, then the run, is the whole guarantee.

    This is the test that caught the first implementation: it queued instead of
    waking waiters to race, but the *decision* was still won by racing the lock,
    so a budget of one still let six prompts through first.
    """
    gate, order = queued_run(batch_yields=1, prompt_count=6)
    assert order == ["prompt", "propagation"] + ["prompt"] * 5, order
    assert not gate.busy and gate.waiting == 0, "the gate leaked its state"


def test_the_deferral_budget_can_be_switched_off():
    """`batch_yields=0` means "let no prompts ahead": the run goes first.

    The boundary of the budget, and a configuration worth having: it is the right
    choice when propagation latency matters more than click latency. It also shows
    the budget is monotonic in one direction — 0 puts the run first, 1 puts one
    prompt in front of it, and so on.
    """
    gate, order = queued_run(batch_yields=0, prompt_count=4)
    assert order == ["propagation"] + ["prompt"] * 4, order
    assert not gate.busy and gate.waiting == 0, "the gate leaked its state"

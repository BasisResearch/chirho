from __future__ import annotations

import heapq
import warnings
from typing import Generic, List, Optional, TypeVar

import pyro

from chirho.dynamical.handlers.interruption import StaticInterruption
from chirho.dynamical.internals._utils import Prioritized
from chirho.dynamical.internals.solver import (
    Interruption,
    apply_interruptions,
    get_new_interruptions,
    get_next_interruptions,
    get_solver,
    simulate_to_interruption,
)

S = TypeVar("S")
T = TypeVar("T")


class InterruptionEventLoop(Generic[T], pyro.poutine.messenger.Messenger):
    _interruption: Optional[Interruption]
    _interruption_stack: List[Interruption]

    def _pyro_simulate(self, msg) -> None:
        dynamics, state, start_time, end_time = msg["args"]
        if msg["kwargs"].get("solver", None) is not None:
            solver = msg["kwargs"]["solver"]
        else:
            solver = get_solver()

        # local state
        active_interruptions: List[Prioritized] = []
        self._interruption_stack = []
        self._interruption = None
        self._start_time = start_time

        heapq.heappush(
            active_interruptions,
            Prioritized(float(end_time), StaticInterruption(end_time)),
        )

        # Simulate through the timespan, stopping at each interruption. This gives e.g. intervention handlers
        #  a chance to modify the state and/or dynamics before the next span is simulated.
        while self._start_time < end_time:
            new_interruptons = get_new_interruptions()
            for h in new_interruptons:
                if isinstance(h, StaticInterruption) and h.time >= end_time:
                    warnings.warn(
                        f"{StaticInterruption.__name__} {h} with time={h.time} "
                        f"occurred after the end of the timespan ({start_time}, {end_time})."
                        "This interruption will have no effect.",
                        UserWarning,
                    )
                elif isinstance(h, StaticInterruption) and h.time < start_time:
                    raise ValueError(
                        f"{StaticInterruption.__name__} {h} with time {h.time} "
                        f"occurred before the start of the timespan ({start_time}, {end_time})."
                        "This interruption will have no effect."
                    )
                else:
                    heapq.heappush(
                        active_interruptions,
                        Prioritized(float(getattr(h, "time", start_time - 1)), h),
                    )

            while active_interruptions:
                ph = heapq.heappop(active_interruptions)
                self._interruption_stack.append(ph.item)
                if ph.priority > self._start_time:
                    break

            state = simulate_to_interruption(
                solver,
                dynamics,
                state,
                self._start_time,
                end_time,
            )

            if self._interruption is not None:
                with self._interruption:
                    dynamics, state = apply_interruptions(dynamics, state)

                while self._interruption_stack:
                    h = self._interruption_stack.pop()
                    if h is not self._interruption:
                        heapq.heappush(
                            active_interruptions,
                            Prioritized(float(getattr(h, "time", start_time - 1)), h),
                        )

                self._interruption = None

        msg["value"] = state
        msg["done"] = True

    def _pyro_simulate_to_interruption(self, msg) -> None:
        solver, dynamics, state, start_time = msg["args"][:-1]

        interruptions = self._interruption_stack

        (self._interruption,), self._start_time = get_next_interruptions(
            solver, dynamics, state, start_time, interruptions
        )

        msg["args"] = msg["args"][:-1] + (self._start_time,)

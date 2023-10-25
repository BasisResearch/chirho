from __future__ import annotations

import warnings
from typing import Generic, List, Optional, TypeVar

import pyro

from chirho.dynamical.handlers.interruption import Interruption, StaticInterruption
from chirho.dynamical.internals.solver import (
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
        self._interruption_stack = [StaticInterruption(end_time)]
        self._interruption = None
        self._start_time = start_time

        # Simulate through the timespan, stopping at each interruption. This gives e.g. intervention handlers
        #  a chance to modify the state and/or dynamics before the next span is simulated.
        while self._start_time < end_time:
            new_interruptons = get_new_interruptions()
            for h in new_interruptons:
                if isinstance(h, StaticInterruption) and not (
                    start_time < h.time < end_time
                ):
                    warnings.warn(
                        f"{StaticInterruption.__name__} {h} with time={h.time} "
                        f"occurred outside the timespan ({start_time}, {end_time})."
                        "This interruption will have no effect.",
                        UserWarning,
                    )
                self._interruption_stack.append(h)

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

                ix = self._interruption_stack.index(self._interruption)
                self._interruption_stack.pop(ix)
                self._interruption = None

        msg["value"] = state
        msg["done"] = True

    def _pyro_simulate_to_interruption(self, msg) -> None:
        solver, dynamics, state, start_time = msg["args"][:-1]

        static_interruptions, dynamic_interruptions = [], []
        for h in self._interruption_stack:
            if isinstance(h, StaticInterruption):
                static_interruptions.append(h)
            else:
                dynamic_interruptions.append(h)

        dynamic_interruptions += [
            min(static_interruptions, key=lambda h: float(h.time))
        ]

        (self._interruption,), self._start_time = get_next_interruptions(
            solver, dynamics, state, start_time, dynamic_interruptions
        )

        msg["args"] = msg["args"][:-1] + (self._start_time,)

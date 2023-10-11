from __future__ import annotations

from typing import Generic, TypeVar

import pyro

from chirho.dynamical.handlers.interruption.interruption import Interruption
from chirho.dynamical.internals.backend import (
    apply_interruptions,
    simulate_to_interruption,
)

S = TypeVar("S")
T = TypeVar("T")


class SimulatorEventLoop(Generic[T], pyro.poutine.messenger.Messenger):
    def _pyro_simulate(self, msg) -> None:
        dynamics, state, start_time, end_time = msg["args"]
        if "solver" in msg["kwargs"]:
            solver = msg["kwargs"]["solver"]
        else:  # Early return to trigger `simulate` ValueError for not having a solver.
            return

        # Simulate through the timespan, stopping at each interruption. This gives e.g. intervention handlers
        #  a chance to modify the state and/or dynamics before the next span is simulated.
        while start_time < end_time:
            with pyro.poutine.messenger.block_messengers(
                lambda m: m is self or (isinstance(m, Interruption) and m.used)
            ):
                state, terminal_interruptions, start_time = simulate_to_interruption(
                    solver,
                    dynamics,
                    state,
                    start_time,
                    end_time,
                )
                for h in terminal_interruptions:
                    h.used = True

            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, Interruption)
                and m not in terminal_interruptions
            ):
                dynamics, state = apply_interruptions(dynamics, state)

        msg["value"] = state
        msg["stop"] = True
        msg["done"] = True
        msg["in_SEL"] = True

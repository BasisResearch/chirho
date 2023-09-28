from __future__ import annotations

import warnings
from typing import Dict, Generic, Tuple, TypeVar

import pyro

from chirho.dynamical.handlers.interruption import (
    DynamicInterruption,
    Interruption,
    PointInterruption,
)
from chirho.dynamical.internals.interruption import (
    apply_interruptions,
    simulate_to_interruption,
)

S = TypeVar("S")
T = TypeVar("T")


class SimulatorEventLoop(Generic[T], pyro.poutine.messenger.Messenger):
    # noinspection PyMethodMayBeStatic
    def _pyro_simulate(self, msg) -> None:
        dynamics, initial_state, start_time, end_time = msg["args"]
        if "solver" in msg["kwargs"]:
            solver = msg["kwargs"]["solver"]
        else:
            # Early return to trigger `simulate` ValueError for not having a solver.
            return

        # Initial values. These will be updated in the loop below.
        span_start_state = initial_state
        span_start_time = start_time

        # We use interruption mechanics to stop the timespan at the right point.
        default_terminal_interruption = PointInterruption(
            time=end_time,
        )

        previous_terminal_interruptions: Tuple[Interruption, ...] = tuple()
        interruption_counts: Dict[Interruption, int] = dict()

        # Simulate through the timespan, stopping at each interruption. This gives e.g. intervention handlers
        #  a chance to modify the state and/or dynamics before the next span is simulated.
        while True:
            # Block any interruption's application that wouldn't be the result of an interruption that ended the last
            #  simulation.
            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, Interruption)
                and m not in previous_terminal_interruptions
            ):
                dynamics, span_start_state = apply_interruptions(
                    dynamics, span_start_state
                )

            # Block dynamic interventions that have triggered and applied more than the specified number of times.
            # This will prevent them from percolating up to the simulate_to_interruption execution.
            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, DynamicInterruption)
                and m.max_applications <= interruption_counts.get(m, 0)
            ):
                (
                    end_state,
                    terminal_interruptions,
                    interruption_time,
                ) = simulate_to_interruption(  # This call gets handled by interruption handlers.
                    dynamics,
                    span_start_state,
                    span_start_time,
                    end_time,
                    solver=solver,
                    # Here, we pass the default terminal interruption — the end of the timespan. Other point/static
                    #  interruption handlers may replace this with themselves if they happen before the end.
                    next_static_interruption=default_terminal_interruption,
                    # We just pass nothing here, as any interruption handlers will be responsible for
                    #  accruing themselves to the message. Leaving explicit for documentation.
                    dynamic_interruptions=None,
                )

            if len(terminal_interruptions) > 1:
                warnings.warn(
                    "Multiple events fired simultaneously. This results in undefined behavior.",
                    UserWarning,
                )

            for interruption in terminal_interruptions:
                interruption_counts[interruption] = (
                    interruption_counts.get(interruption, 0) + 1
                )

            # If we've reached the end of the timespan, break.
            if default_terminal_interruption in terminal_interruptions:
                break

            # Set the span_start_time for the next iteration to be the interruption time from the previous.
            # TODO AZ — we should be able to detect when this eps is too small, as it will repeatedly trigger
            #  the same event at the same time.
            span_start_time = interruption_time

            # Update the starting state.
            span_start_state = end_state

            # Use these to block interruption handlers that weren't responsible for the last interruption.
            previous_terminal_interruptions = terminal_interruptions

        msg["value"] = end_state
        msg["done"] = True

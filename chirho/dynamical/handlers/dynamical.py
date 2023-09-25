from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, Generic, List, Tuple, TypeVar

import pyro
import torch

from chirho.dynamical.handlers.interruption import (
    DynamicInterruption,
    Interruption,
    PointInterruption,
)
from chirho.dynamical.internals.interruption import (
    apply_interruptions,
    concatenate,
    simulate_to_interruption,
)

if TYPE_CHECKING:
    from chirho.dynamical.ops import State

from chirho.dynamical.ops import Trajectory

S = TypeVar("S")
T = TypeVar("T")


class SimulatorEventLoop(Generic[T], pyro.poutine.messenger.Messenger):
    def __enter__(self):
        return super().__enter__()

    # noinspection PyMethodMayBeStatic
    def _pyro_simulate(self, msg) -> None:
        dynamics, initial_state, full_timespan = msg["args"]
        if "backend" in msg["kwargs"]:
            backend = msg["kwargs"]["backend"]
        else:
            # Early return to trigger `simulate` ValueError for not having a backend.
            return

        # Initial values. These will be updated in the loop below.
        span_start_state = initial_state
        span_timespan = full_timespan

        # We use interruption mechanics to stop the timespan at the right point.
        default_terminal_interruption = PointInterruption(
            time=span_timespan[-1],
        )

        full_trajs: List[Trajectory[T]] = []
        first = True

        last_terminal_interruptions: Tuple[Interruption, ...] = tuple()
        interruption_counts: Dict[Interruption, int] = dict()

        # Simulate through the timespan, stopping at each interruption. This gives e.g. intervention handlers
        #  a chance to modify the state and/or dynamics before the next span is simulated.
        while True:
            # Block any interruption's application that wouldn't be the result of an interruption that ended the last
            #  simulation.
            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, Interruption)
                and m not in last_terminal_interruptions
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
                    span_traj,
                    terminal_interruptions,
                    end_time,
                    end_state,
                ) = simulate_to_interruption(  # This call gets handled by interruption handlers.
                    dynamics,
                    span_start_state,
                    span_timespan,
                    backend=backend,
                    # Here, we pass the default terminal interruption — the end of the timespan. Other point/static
                    #  interruption handlers may replace this with themselves if they happen before the end.
                    next_static_interruption=default_terminal_interruption,
                    # We just pass nothing here, as any interruption handlers will be responsible for
                    #  accruing themselves to the message. Leaving explicit for documentation.
                    dynamic_interruptions=None,
                )  # type: Trajectory[T], Tuple['Interruption', ...], torch.Tensor, State[T]

            if len(terminal_interruptions) > 1:
                warnings.warn(
                    "Multiple events fired simultaneously. This results in undefined behavior.",
                    UserWarning,
                )

            for interruption in terminal_interruptions:
                interruption_counts[interruption] = (
                    interruption_counts.get(interruption, 0) + 1
                )

            last = default_terminal_interruption in terminal_interruptions

            # Update the full trajectory.
            if first:
                full_trajs.append(span_traj)
            else:
                # Hack off the end time of the previous simulate_to_interruption, as the user didn't request this.
                # if any(s == 0 for k in span_traj.keys for s in getattr(span_traj[..., 1:], k).shape):
                #     full_trajs.append(span_traj[..., 1:])
                # TODO support event_dim > 0
                span_traj_: Trajectory[T] = span_traj[..., 1:]
                full_trajs.append(span_traj_)

            # If we've reached the end of the timespan, break.
            if last:
                # The end state in this case will be the final tspan requested by the user, so we need to include.
                # TODO support event_dim > 0
                full_trajs.append(end_state.trajectorify())
                break

            # Construct the next timespan so that we simulate from the prevous interruption time.
            # TODO AZ — we should be able to detect when this eps is too small, as it will repeatedly trigger
            #  the same event at the same time.
            span_timespan = torch.cat(
                (end_time.unsqueeze(0), full_timespan[full_timespan > end_time])
            )

            # Update the starting state.
            span_start_state = end_state

            # Use these to block interruption handlers that weren't responsible for the last interruption.
            last_terminal_interruptions = terminal_interruptions

            first = False

        msg["value"] = concatenate(*full_trajs)
        msg["done"] = True

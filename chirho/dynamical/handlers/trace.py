from __future__ import annotations

from typing import Generic, TypeVar

import pyro
import torch

from chirho.dynamical.internals.dynamical import simulate_trajectory
from chirho.dynamical.ops import Trajectory, State

T = TypeVar("T")


class DynamicTrace(Generic[T], pyro.poutine.messenger.Messenger):
    def __init__(self, logging_times: T, epsilon: float = 1e-6):
        # Adding epsilon to the logging times to avoid collision issues with the logging times being exactly on the
        #  boundaries of the simulation times. This is a hack, but it's a hack that should work for now.
        self.logging_times = logging_times + epsilon
        self._reset()
        super().__init__()

    def _reset(self):
        self.trace = Trajectory()

    # TODO: Pick up here.
    def _pyro_simulate(self, msg) -> None:
        # Turn a simulate that returns a state into a simulate that returns a trajectory at each of the logging_times
        dynamics, initial_state, start_time, end_time = msg["args"]
        if "solver" in msg["kwargs"]:
            solver = msg["kwargs"]["solver"]
        else:
            # Early return to trigger `simulate` ValueError for not having a solver.
            return

        filtered_timespan = self.logging_times[
            (self.logging_times >= start_time) & (self.logging_times <= end_time)
        ]
        timespan = torch.concat(
            (start_time.unsqueeze(-1), filtered_timespan, end_time.unsqueeze(-1))
        )

        trajectory = simulate_trajectory(dynamics, initial_state, timespan, solver=solver)
        self.trace.append(trajectory[1:-1])
        # TODO: check to make sure we don't need leading ... dimension. E.g. `trajectory[..., -1]`
        msg["value"] = trajectory[-1]
        msg["done"] = True

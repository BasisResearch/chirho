from typing import Generic, TypeVar

import pyro
import torch

from chirho.dynamical.internals.dynamical import simulate_trajectory
from chirho.dynamical.ops import Trajectory

T = TypeVar("T")


class DynamicTrace(Generic[T], pyro.poutine.messenger.Messenger):
    def __init__(self, logging_times: torch.Tensor, epsilon: float = 1e-6):
        # Adding epsilon to the logging times to avoid collision issues with the logging times being exactly on the
        #  boundaries of the simulation times. This is a hack, but it's a hack that should work for now.
        self.logging_times = logging_times + epsilon
        self._reset()

        # Require that the times are sorted. This is required by the index masking we do below.
        # TODO AZ sort this here (and the data too) accordingly?
        if not torch.all(self.logging_times[1:] > self.logging_times[:-1]):
            raise ValueError("The passed times must be sorted.")

        super().__init__()

    def _reset(self):
        self.trace = Trajectory()

    def _pyro_simulate(self, msg) -> None:
        msg["done"] = True

    def _pyro_post_simulate(self, msg) -> None:
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

        trajectory = simulate_trajectory(
            solver,
            dynamics,
            initial_state,
            timespan,
        )
        self.trace.append(trajectory[..., 1:-1])
        if len(self.trace) > len(self.logging_times):
            raise ValueError(
                "Multiple simulates were used with a single DynamicTrace handler."
                "This is currently not supported."
            )
        msg["value"] = trajectory[..., -1].to_state()
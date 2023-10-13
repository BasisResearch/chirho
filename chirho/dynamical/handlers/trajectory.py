import typing
from typing import Generic, TypeVar

import pyro
import torch

from chirho.dynamical.internals._utils import append
from chirho.dynamical.internals.solver import Solver, get_solver, simulate_trajectory
from chirho.dynamical.ops import Trajectory

T = TypeVar("T")


class LogTrajectory(Generic[T], pyro.poutine.messenger.Messenger):
    trajectory: Trajectory[T]

    def __init__(self, times: torch.Tensor, *, eps: float = 1e-6):
        # Adding epsilon to the logging times to avoid collision issues with the logging times being exactly on the
        #  boundaries of the simulation times. This is a hack, but it's a hack that should work for now.
        self.times = times + eps

        # Require that the times are sorted. This is required by the index masking we do below.
        if not torch.all(self.times[1:] > self.times[:-1]):
            raise ValueError("The passed times must be sorted.")

        super().__init__()

    def __enter__(self) -> "LogTrajectory[T]":
        self.trajectory: Trajectory[T] = Trajectory()
        return super().__enter__()

    def _pyro_simulate(self, msg) -> None:
        msg["done"] = True

    def _pyro_post_simulate(self, msg) -> None:
        # Turn a simulate that returns a state into a simulate that returns a trajectory at each of the logging_times
        dynamics, initial_state, start_time, end_time = msg["args"]
        if msg["kwargs"].get("solver", None) is not None:
            solver = typing.cast(Solver, msg["kwargs"]["solver"])
        else:
            solver = get_solver()

        filtered_timespan = self.times[
            (self.times >= start_time) & (self.times <= end_time)
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
        idx = (timespan > timespan[0]) & (timespan < timespan[-1])
        if idx.any():
            self.trajectory: Trajectory[T] = append(self.trajectory, trajectory[idx])
        if idx.sum() > len(self.times):
            raise ValueError(
                "Multiple simulates were used with a single LogTrajectory handler."
                "This is currently not supported."
            )
        msg["value"] = trajectory[timespan == timespan[-1]].to_state()

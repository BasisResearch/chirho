import typing
from typing import Generic, TypeVar

import pyro
import torch

from chirho.dynamical.internals._utils import _trajectory_to_state, append
from chirho.dynamical.internals.solver import Solver, get_solver, simulate_trajectory
from chirho.dynamical.ops import State
from chirho.indexed.ops import IndexSet, gather, get_index_plates

T = TypeVar("T")


class LogTrajectory(Generic[T], pyro.poutine.messenger.Messenger):
    trajectory: State[T]

    def __init__(self, times: torch.Tensor, *, eps: float = 1e-6):
        # Adding epsilon to the logging times to avoid collision issues with the logging times being exactly on the
        #  boundaries of the simulation times. This is a hack, but it's a hack that should work for now.
        self.times = times + eps

        # Require that the times are sorted. This is required by the index masking we do below.
        if not torch.all(self.times[1:] > self.times[:-1]):
            raise ValueError("The passed times must be sorted.")

        super().__init__()

    def __enter__(self) -> "LogTrajectory[T]":
        self.trajectory: State[T] = State()
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

        # TODO support dim != -1
        idx_name = "__time"
        name_to_dim = {k: f.dim - 1 for k, f in get_index_plates().items()}
        name_to_dim[idx_name] = -1

        if len(timespan) > 2:
            part_idx = IndexSet(**{idx_name: set(range(1, len(timespan) - 1))})
            new_part = gather(trajectory, part_idx, name_to_dim=name_to_dim)
            self.trajectory: State[T] = append(self.trajectory, new_part)

        final_idx = IndexSet(**{idx_name: {len(timespan) - 1}})
        final_state = gather(trajectory, final_idx, name_to_dim=name_to_dim)
        msg["value"] = _trajectory_to_state(final_state)

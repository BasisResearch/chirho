from typing import Generic, TypeVar

import pyro
import torch

from chirho.dynamical.internals._utils import _squeeze_time_dim, append
from chirho.dynamical.internals.solver import get_solver, simulate_trajectory
from chirho.dynamical.ops import State
from chirho.indexed.ops import IndexSet, gather, get_index_plates

T = TypeVar("T")


class LogTrajectory(Generic[T], pyro.poutine.messenger.Messenger):
    trajectory: State[T]

    def __init__(self, times: torch.Tensor):
        self.times = times

        # Require that the times are sorted. This is required by the index masking we do below.
        if not torch.all(self.times[1:] > self.times[:-1]):
            raise ValueError("The passed times must be sorted.")

        super().__init__()

    def __enter__(self) -> "LogTrajectory[T]":
        self.trajectory: State[T] = State()
        return super().__enter__()

    def _pyro_simulate_point(self, msg) -> None:
        # Turn a simulate that returns a state into a simulate that returns a trajectory at each of the logging_times
        _, dynamics, initial_state, start_time, end_time = msg["args"]

        filtered_timespan = self.times[
            (self.times >= start_time) & (self.times <= end_time)
        ]
        timespan = torch.concat(
            (start_time.unsqueeze(-1), filtered_timespan, end_time.unsqueeze(-1))
        )

        with pyro.poutine.messenger.block_messengers(lambda m: m is self):
            trajectory = simulate_trajectory(
                get_solver(),
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
        msg["value"] = _squeeze_time_dim(
            gather(trajectory, final_idx, name_to_dim=name_to_dim)
        )
        msg["done"] = True

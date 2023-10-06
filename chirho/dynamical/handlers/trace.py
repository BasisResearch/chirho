import functools
from typing import Generic, TypeVar

import pyro
import torch

from chirho.dynamical.internals.dynamical import simulate_trajectory
from chirho.dynamical.ops import Trajectory

T = TypeVar("T")


@functools.singledispatch
def append(fst, rest: T) -> T:
    raise NotImplementedError(f"append not implemented for type {type(fst)}.")


@append.register(Trajectory)
def append_trajectory(traj1: Trajectory[T], traj2: Trajectory[T]) -> Trajectory[T]:
    if len(traj1.keys) == 0:
        return traj2

    if len(traj2.keys) == 0:
        return traj1

    if traj1.keys != traj2.keys:
        raise ValueError(
            f"Trajectories must have the same keys to be appended, but got {traj1.keys} and {traj2.keys}."
        )

    result: Trajectory[T] = Trajectory()
    for k in traj1.keys:
        setattr(result, k, append(getattr(traj1, k), getattr(traj2, k)))

    return result


@append.register(torch.Tensor)
def append_tensor(prev_v: torch.Tensor, curr_v: torch.Tensor) -> torch.Tensor:
    time_dim = -1  # TODO generalize to nontrivial event_shape
    batch_shape = torch.broadcast_shapes(prev_v.shape[:-1], curr_v.shape[:-1])
    prev_v = prev_v.expand(*batch_shape, *prev_v.shape[-1:])
    curr_v = curr_v.expand(*batch_shape, *curr_v.shape[-1:])
    return torch.cat([prev_v, curr_v], dim=time_dim)


class DynamicTrace(Generic[T], pyro.poutine.messenger.Messenger):
    trace: Trajectory[T]

    def __init__(self, logging_times: torch.Tensor, epsilon: float = 1e-6):
        self.trace: Trajectory[T] = Trajectory()

        # Adding epsilon to the logging times to avoid collision issues with the logging times being exactly on the
        #  boundaries of the simulation times. This is a hack, but it's a hack that should work for now.
        self.logging_times = logging_times + epsilon

        # Require that the times are sorted. This is required by the index masking we do below.
        # TODO AZ sort this here (and the data too) accordingly?
        if not torch.all(self.logging_times[1:] > self.logging_times[:-1]):
            raise ValueError("The passed times must be sorted.")

        super().__init__()

    def __enter__(self):
        self.trace: Trajectory[T] = Trajectory()
        return super().__enter__()

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
        self.trace: Trajectory[T] = append(self.trace, trajectory[..., 1:-1])
        if len(self.trace) > len(self.logging_times):
            raise ValueError(
                "Multiple simulates were used with a single DynamicTrace handler."
                "This is currently not supported."
            )
        msg["value"] = trajectory[..., -1].to_state()

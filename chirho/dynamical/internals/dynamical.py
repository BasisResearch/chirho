import functools
from typing import Optional, TypeVar

import torch

from chirho.dynamical.handlers.solver import Solver
from chirho.dynamical.ops.dynamical import Dynamics, State, Trajectory

S = TypeVar("S")
T = TypeVar("T")


@functools.singledispatch
def simulate_trajectory(
    dynamics: Dynamics[S, T],
    initial_state: State[T],
    timespan: T,
    *,
    solver: Optional[Solver] = None,
    **kwargs,
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    if solver is None:
        raise ValueError(
            "`simulate_trajectory` requires a solver. To specify a solver, use the keyword argument `solver` in"
            " the call to `simulate_trajectory` or use with a solver effect handler as a context manager. For example, \n \n"
            "`with TorchDiffEq():` \n"
            "\t `simulate_trajectory(dynamics, initial_state, start_time, end_time)`"
        )
    raise NotImplementedError(
        f"simulate_trajectory not implemented for type {type(dynamics)}"
    )


@functools.singledispatch
def unsqueeze(x, axis: int):
    raise NotImplementedError(f"unsqueeze not implemented for type {type(x)}")


@unsqueeze.register
def _unsqueeze_torch(x: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.unsqueeze(x, axis)

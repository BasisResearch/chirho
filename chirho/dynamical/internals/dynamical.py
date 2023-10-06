from __future__ import annotations

import functools
from typing import TYPE_CHECKING, TypeVar

import torch

from chirho.dynamical.ops.dynamical import Dynamics, State, Trajectory

if TYPE_CHECKING:
    from chirho.dynamical.handlers.solver import Solver


S = TypeVar("S")
T = TypeVar("T")


@functools.singledispatch
def simulate_trajectory(
    solver: "Solver",  # Quoted type necessary w/ TYPE_CHECKING to avoid circular import error
    dynamics: Dynamics[S, T],
    initial_state: State[T],
    timespan: T,
    **kwargs,
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(
        f"simulate_trajectory not implemented for solver of type {type(solver)}"
    )


@functools.singledispatch
def unsqueeze(x, axis: int):
    raise NotImplementedError(f"unsqueeze not implemented for type {type(x)}")


@unsqueeze.register
def _unsqueeze_torch(x: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.unsqueeze(x, axis)

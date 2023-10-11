from __future__ import annotations

import functools
from typing import TYPE_CHECKING, TypeVar

from chirho.dynamical.ops.dynamical import InPlaceDynamics, State, Trajectory

if TYPE_CHECKING:
    from chirho.dynamical.handlers.solver import Solver


S = TypeVar("S")
T = TypeVar("T")


@functools.singledispatch
def simulate_trajectory(
    solver: "Solver",  # Quoted type necessary w/ TYPE_CHECKING to avoid circular import error
    dynamics: InPlaceDynamics[T],
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

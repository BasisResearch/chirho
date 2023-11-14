import numbers
import sys
import typing
from typing import Callable, Dict, Generic, Optional, TypeVar, Union

import pyro
import torch

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


if typing.TYPE_CHECKING:
    State = Dict[str, T]
elif sys.version_info >= (3, 9):
    State = dict[str, T]
else:

    class State(Generic[T], Dict[str, T]):
        pass


Dynamics = Callable[[State[T]], State[T]]


@pyro.poutine.runtime.effectful(type="simulate")
def simulate(
    dynamics: Dynamics[T],
    initial_state: State[T],
    start_time: R,
    end_time: R,
    *,
    solver: Optional[S] = None,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    from chirho.dynamical.internals.solver import (
        Solver,
        get_solver,
        simulate_to_interruption,
    )

    solver_: Solver = get_solver() if solver is None else typing.cast(Solver, solver)
    return simulate_to_interruption(
        solver_, dynamics, initial_state, start_time, end_time, **kwargs
    )

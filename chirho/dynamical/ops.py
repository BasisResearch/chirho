import numbers
import typing
from typing import Callable, Dict, Generic, Optional, TypeVar, Union

import pyro
import torch

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


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
        simulate_point,
    )

    if solver is None:
        return simulate_point(
            get_solver(), dynamics, initial_state, start_time, end_time, **kwargs
        )
    else:
        assert isinstance(solver, Solver)
        with solver:
            return simulate_point(
                solver, dynamics, initial_state, start_time, end_time, **kwargs
            )

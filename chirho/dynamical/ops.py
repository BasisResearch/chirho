import contextlib
import numbers
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
    solver: Optional[pyro.poutine.messenger.Messenger] = None,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    from chirho.dynamical.internals.solver import simulate_point

    with contextlib.nullcontext() if solver is None else solver:
        return simulate_point(dynamics, initial_state, start_time, end_time, **kwargs)

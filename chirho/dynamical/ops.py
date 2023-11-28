import numbers
import sys
import typing
from typing import Callable, Dict, Generic, TypeVar, Union

import pyro
import torch

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


if typing.TYPE_CHECKING:
    State = Dict[str, T]
elif sys.version_info >= (3, 9):
    State = dict
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
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    from chirho.dynamical.internals.solver import check_dynamics, simulate_point

    if pyro.settings.get("validate_dynamics"):
        check_dynamics(dynamics, initial_state, start_time, end_time, **kwargs)
    return simulate_point(dynamics, initial_state, start_time, end_time, **kwargs)

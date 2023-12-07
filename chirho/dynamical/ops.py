import numbers
from typing import Callable, Mapping, Optional, Tuple, TypeVar, Union

import pyro
import torch

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")

State = Mapping[str, T]
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


def on(
    predicate: Callable[[State[T]], bool],
    callback: Optional[
        Callable[[Dynamics[T], State[T]], Tuple[Dynamics[T], State[T]]]
    ] = None,
):
    if callback is None:

        def _on(
            callback: Callable[[Dynamics[T], State[T]], Tuple[Dynamics[T], State[T]]]
        ):
            return on(predicate, callback)

        return _on

    from chirho.dynamical.internals.solver import Interruption

    return Interruption(predicate, callback)

import contextlib
import functools
import numbers
from typing import (
    Callable,
    Concatenate,
    Mapping,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
)

import pyro
import torch

R = Union[numbers.Real, torch.Tensor]
P = ParamSpec("P")
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
    predicate: Callable[Concatenate[State[T], P], bool],
    callback: Optional[
        Callable[Concatenate[Dynamics[T], State[T], P], Tuple[Dynamics[T], State[T]]]
    ] = None,
):
    if callback is None:
        return functools.partial(on, predicate)

    from chirho.dynamical.internals.solver import Interruption

    @contextlib.contextmanager
    def cm(*args: P.args, **kwargs: P.kwargs):
        predicate_: Callable[[State[T]], bool] = lambda state: predicate(
            state, *args, **kwargs
        )
        callback_: Callable[
            [Dynamics[T], State[T]], Tuple[Dynamics[T], State[T]]
        ] = lambda dynamics, state: callback(dynamics, state, *args, **kwargs)
        with Interruption(predicate_, callback_):
            yield

    return cm

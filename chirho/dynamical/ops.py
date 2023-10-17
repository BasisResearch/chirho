import numbers
import typing
from typing import FrozenSet, Generic, Optional, Protocol, TypeVar, Union

import pyro
import torch

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class State(Generic[T]):
    def __init__(self, time: Optional[T] = None, **values: T):
        self.__dict__["_values"] = {}
        if time is not None:
            self.time = time
        for k, v in values.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__['_values']})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__['_values']})"

    def __setattr__(self, __name: str, __value: T) -> None:
        self.__dict__["_values"][__name] = __value

    def __getattr__(self, __name: str) -> T:
        if __name in self.__dict__["_values"]:
            return self.__dict__["_values"][__name]
        else:
            raise AttributeError(f"{__name} not in {self.__dict__['_values']}")


def get_keys(state: State[T], include_time: bool = True) -> FrozenSet[str]:
    if include_time:
        return frozenset(state.__dict__["_values"].keys())
    else:
        return frozenset(k for k in state.__dict__["_values"].keys() if k != "time")


@typing.runtime_checkable
class Observable(Protocol[S]):
    def observation(self, __state: State[S]) -> None:
        ...


@typing.runtime_checkable
class InPlaceDynamics(Protocol[S]):
    def diff(self, __dstate: State[S], __state: State[S]) -> None:
        ...


@pyro.poutine.runtime.effectful(type="simulate")
def simulate(
    dynamics: InPlaceDynamics[T],
    initial_state: State[T],
    end_time: R,
    *,
    solver: Optional[S] = None,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    from chirho.dynamical.internals.solver import Solver, get_solver, simulate_point

    solver_: Solver = get_solver() if solver is None else typing.cast(Solver, solver)
    return simulate_point(solver_, dynamics, initial_state, end_time, **kwargs)

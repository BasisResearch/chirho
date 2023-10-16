import numbers
import typing
from typing import FrozenSet, Generic, Optional, Protocol, TypeVar, Union

import pyro
import torch

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class State(Generic[T]):
    def __init__(self, **values: T):
        self.__dict__["_values"] = {}
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


class Trajectory(Generic[T], State[T]):
    pass


def get_keys(state: State[T]) -> FrozenSet[str]:
    return frozenset(state.__dict__["_values"].keys())


@typing.runtime_checkable
class InPlaceDynamics(Protocol[S]):
    def diff(self, __dstate: State[S], __state: State[S]) -> None:
        ...


@typing.runtime_checkable
class ObservableInPlaceDynamics(InPlaceDynamics[S], Protocol[S]):
    def diff(self, __dstate: State[S], __state: State[S]) -> None:
        ...

    def observation(self, __state: Union[State[S], Trajectory[S]]) -> None:
        ...


@pyro.poutine.runtime.effectful(type="simulate")
def simulate(
    dynamics: InPlaceDynamics[T],
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
    from chirho.dynamical.internals.solver import Solver, get_solver, simulate_point

    solver_: Solver = get_solver() if solver is None else typing.cast(Solver, solver)
    return simulate_point(
        solver_, dynamics, initial_state, start_time, end_time, **kwargs
    )

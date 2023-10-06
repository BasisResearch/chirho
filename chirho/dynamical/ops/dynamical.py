import functools
from typing import (
    TYPE_CHECKING,
    Callable,
    FrozenSet,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import pyro

if TYPE_CHECKING:
    from chirho.dynamical.handlers.solver import Solver

S = TypeVar("S")
T = TypeVar("T")


class State(Generic[T]):
    def __init__(self, **values: T):
        self.__dict__["_values"] = {}
        for k, v in values.items():
            setattr(self, k, v)

    @property
    def var_order(self):
        return tuple(sorted(self.keys))

    @property
    def keys(self) -> FrozenSet[str]:
        return frozenset(self.__dict__["_values"].keys())

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


@runtime_checkable
class Dynamics(Protocol[S, T]):
    diff: Callable[[State[S], State[S]], T]
    observation: Callable[[State[S]], None]


@pyro.poutine.runtime.effectful(type="simulate")
def simulate(
    dynamics: Dynamics[S, T],
    initial_state: State[T],
    start_time: T,
    end_time: T,
    *,
    solver: Optional[
        "Solver"
    ] = None,  # Quoted type necessary w/ TYPE_CHECKING to avoid circular import error
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    if solver is None:
        raise ValueError(
            "`simulate`` requires a solver. To specify a solver, use the keyword argument `solver` in"
            " the call to `simulate` or use with a solver effect handler as a context manager. For example, \n \n"
            "`with SimulatorEventLoop():` \n"
            "\t `with TorchDiffEq():` \n"
            "\t \t `simulate(dynamics, initial_state, start_time, end_time)`"
        )
    return _simulate(solver, dynamics, initial_state, start_time, end_time, **kwargs)


# This redirection distinguishes between the effectful operation, and the
# type-directed dispatch on Dynamics
@functools.singledispatch
def _simulate(
    solver: "Solver",  # Quoted type necessary w/ TYPE_CHECKING to avoid circular import error
    dynamics: Dynamics[S, T],
    initial_state: State[T],
    start_time: T,
    end_time: T,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(
        f"simulate not implemented for solver of type {type(solver)}"
    )


simulate.register = _simulate.register

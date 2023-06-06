from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import functools
import pyro
import torch

S, T = TypeVar("S"), TypeVar("T")


class State(Generic[T]):
    def __init__(self, **values: T):
        self.__dict__["_values"]: dict[str, T] = {}
        for k, v in values.items():
            setattr(self, k, v)

    @property
    def keys(self) -> Set[str]:
        return frozenset(self.__dict__["_values"].keys())

    def __repr__(self) -> str:
        return f"State({self.__dict__['_values']})"

    def __str__(self) -> str:
        return f"State({self.__dict__['_values']})"

    @pyro.poutine.runtime.effectful(type="state_setattr")
    def __setattr__(self, __name: str, __value: T) -> None:
        self.__dict__["_values"][__name] = __value

    @pyro.poutine.runtime.effectful(type="state_getattr")
    def __getattr__(self, __name: str) -> T:
        if __name in self.__dict__["_values"]:
            return self.__dict__["_values"][__name]
        return super().__getattr__(__name)


class Trajectory(State[T]):
    def __init__(self, **values: T):
        super().__init__(**values)

    def __getitem__(self, item: int) -> State[T]:
        assert isinstance(item, int), "We don't support slicing trajectories."
        state = State()
        for k, v in self.__dict__["_values"].items():
            setattr(state, k, v[item])
        return state


@runtime_checkable
class Dynamics(Protocol[S, T]):
    diff: Callable[[State[S], State[S]], T]


@pyro.poutine.runtime.effectful(type="simulate_step")
def simulate_step(dynamics, curr_state: State[T], dt) -> None:
    pass


@pyro.poutine.runtime.effectful(type="get_dt")
def get_dt(dynamics, curr_state: State[T]):
    pass


@functools.singledispatch
def simulate(
    dynamics: Dynamics[S, T], initial_state: State[T], timespan, **kwargs
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(f"simulate not implemented for type {type(dynamics)}")


@functools.singledispatch
def concatenate(input1, input2):
    """
    Concatenate two Ts.
    """
    raise NotImplementedError(f"concatenate not implemented for type {type(input1)}")


@concatenate.register
def trajectory_concatenate(state1: Trajectory, state2: Trajectory) -> Trajectory:
    """
    Concatenate two states.
    """
    return Trajectory(
        **{
            k: torch.cat([getattr(state1, k)[:-1], getattr(state2, k)[1:]])
            for k in state1.keys
        }
    )

from typing import (
    Any,
    List,
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

    def __getitem__(self, key: int) -> State[T]:
        if isinstance(key, int):
            state = State()
            for k, v in self.__dict__["_values"].keys():
                setattr(state, k, v[key])
            return state
        elif isinstance(key, slice):
            state = State()
            start, end = key
            for k, v in self.__dict__["_values"].keys():
                setattr(state, k, v[start:end])
            return state
        else:
            raise TypeError(f"Expected type int or slice, got {type(key)}")


@runtime_checkable
class Dynamics(Protocol[S, T]):
    diff: Callable[[State[S], State[S]], T]


@functools.singledispatch
def simulate_span(
    dynamics: Dynamics[S, T], curr_state: State[T], timespan, **kwargs
) -> Trajectory[T]:
    """
    Simulate a fixed timespan of a dynamical system.
    """
    raise NotImplementedError(
        f"simulate_span not implemented for type {type(dynamics)}"
    )


@functools.singledispatch
def concatenate(inputs):
    """
    Concatenate multiple inputs of type T into a single output of type T.
    """
    raise NotImplementedError(f"concatenate not implemented for type {type(inputs[0])}")


@concatenate.register
def trajectory_concatenate(trajectories: List[Trajectory]) -> Trajectory:
    """
    Concatenate multiple trajectories into a single trajectory.
    """
    full_trajectory = Trajectory()
    for trajectory in trajectories:
        for k in trajectory.keys:
            if k not in full_trajectory.keys:
                setattr(full_trajectory, k, getattr(trajectory, k))
            else:
                setattr(
                    full_trajectory,
                    k,
                    torch.cat([getattr(full_trajectory, k), getattr(trajectory, k)]),
                )
    return full_trajectory


@functools.singledispatch
def simulate(
    dynamics: Dynamics[S, T], initial_state: State[T], timespan, **kwargs
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(f"simulate not implemented for type {type(dynamics)}")

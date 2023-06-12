import functools
from typing import (
    Callable,
    FrozenSet,
    Generic,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import pyro
import torch

S = TypeVar("S")
T = TypeVar("T")


class State(Generic[T]):
    def __init__(self, **values: T):
        self.__dict__["_values"] = {}
        for k, v in values.items():
            setattr(self, k, v)

    @property
    def keys(self) -> FrozenSet[str]:
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
        else:
            raise AttributeError(f"{__name} not in {self.__dict__['_values']}")


class Trajectory(State[T]):
    def __init__(self, **values: T):
        super().__init__(**values)

    def __getitem__(self, key: Union[int, slice]) -> State[T]:
        if isinstance(key, str):
            raise ValueError(
                "Trajectory does not support string indexing, use getattr instead if you want to access \
                    a specific state variable."
            )

        item: Union[State[T], Trajectory[T]] = (
            State() if isinstance(key, int) else Trajectory()
        )
        for k, v in self.__dict__["_values"].items():
            setattr(item, k, v[key])
        return item


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
def concatenate(*inputs):
    """
    Concatenate multiple inputs of type T into a single output of type T.
    """
    raise NotImplementedError(f"concatenate not implemented for type {type(inputs[0])}")


@concatenate.register
def trajectory_concatenate(*trajectories: Trajectory) -> Trajectory[T]:
    """
    Concatenate multiple trajectories into a single trajectory.
    """
    full_trajectory: Trajectory[T] = Trajectory()
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

from typing import (
    TYPE_CHECKING,
    Callable,
    FrozenSet,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .handlers import DynamicInterruption, PointInterruption, Interruption

from chirho.dynamical.internals import State, Trajectory

import functools

import pyro
import torch

S = TypeVar("S")
T = TypeVar("T")


@runtime_checkable
class Dynamics(Protocol[S, T]):
    diff: Callable[[State[S], State[S]], T]

# noinspection PyUnusedLocal
@functools.singledispatch
def simulate(
    dynamics: Dynamics[S, T], initial_state: State[T], timespan, **kwargs
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(f"simulate not implemented for type {type(dynamics)}")
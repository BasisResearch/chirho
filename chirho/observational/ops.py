from __future__ import annotations

import functools
from collections.abc import Hashable, Mapping
from typing import Callable, TypeVar, Union

T = TypeVar("T")

AtomicObservation = Union[T, Callable[..., T]]  # TODO add support for more atomic types
CompoundObservation = Union[Mapping[Hashable, AtomicObservation[T]], Callable[..., AtomicObservation[T]]]
Observation = Union[AtomicObservation[T], CompoundObservation[T]]


@functools.singledispatch
def observe(rv, obs: Observation[T] | None = None, **kwargs) -> T:
    """
    Observe a random value in a probabilistic program.
    """
    raise NotImplementedError(f"observe not implemented for type {type(rv)}")

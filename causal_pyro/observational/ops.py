import functools
from typing import Callable, Hashable, Mapping, Optional, Tuple, TypeVar, Union

T = TypeVar("T")

AtomicObservation = Union[T, Tuple[T, ...], Callable[[T], Union[T, Tuple[T, ...]]]]
CompoundObservation = Union[Mapping[Hashable, AtomicObservation[T]], Callable[..., T]]
Observation = Union[AtomicObservation[T], CompoundObservation[T]]


@functools.singledispatch
def observe(rv, obs: Optional[Observation[T]] = None, **kwargs):
    """
    Observe a random value in a probabilistic program.
    """
    raise NotImplementedError(f"observe not implemented for type {type(rv)}")

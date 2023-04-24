import functools
from typing import Callable, Hashable, Mapping, Optional, Tuple, TypeVar, Union

T = TypeVar("T")

AtomicIntervention = Union[T, Tuple[T, ...], Callable[[T], T]]
CompoundIntervention = Union[Mapping[Hashable, AtomicIntervention[T]], Callable[..., T]]
Intervention = Union[AtomicIntervention[T], CompoundIntervention[T]]


@functools.singledispatch
def intervene(obs, act: Optional[Intervention[T]] = None, **kwargs):
    """
    Intervene on a value in a probabilistic program.
    """
    raise NotImplementedError(f"intervene not implemented for type {type(obs)}")

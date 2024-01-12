from __future__ import annotations

import functools
from typing import Callable, Hashable, Mapping, Optional, Tuple, TypeVar, Union

T = TypeVar("T")

AtomicIntervention = Union[T, Tuple[T, ...], Callable[[T], Union[T, Tuple[T, ...]]]]
CompoundIntervention = Union[Mapping[Hashable, AtomicIntervention[T]], Callable[..., T]]
Intervention = Union[AtomicIntervention[T], CompoundIntervention[T]]


@functools.singledispatch
def intervene(obs, act: Optional[Intervention[T]] = None, **kwargs):
    """
    Intervene on a value in a probabilistic program.

    :func:`intervene` is primarily used internally in :class:`DoMessenger`
    for concisely and extensibly defining the semantics of interventions. This
    function is generically typed and extensible to new types via
    :func:`functools.singledispatch`. When its first argument is a function,
    :func:`intervene` now behaves like the current `observational.do` effect handler.

    :param obs: a value in a probabilistic program.
    :param act: an optional intervention.
    """
    raise NotImplementedError(f"intervene not implemented for type {type(obs)}")

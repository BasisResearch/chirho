from typing import Callable, Dict, Generic, Hashable, Iterable, List, Optional, Protocol, Set, Tuple, TypeVar, Union

import functools
import multipledispatch
import numbers

import pyro
import torch

R = torch.Tensor | numbers.Number
S = TypeVar("S")
T = TypeVar("T")


class Measure(Protocol[T]):
    base: Optional["Measure[T]"]
    density: Callable[[T], R]
    # transform: Callable[[T], T]


Kernel = Callable[[T], Measure[T]]
MaybeKernel = Measure[T] | Kernel[T]


@functools.singledispatch
def as_measure(x, **kwargs) -> MaybeKernel[T]:
    raise NotImplementedError


@functools.singledispatch
def density(m: Measure[T], other: Optional[Measure[T]] = None) -> Callable[[T], R]:
    return lambda x: density(base_measure(m))(x) * m.density(x)


@functools.singledispatch
def integrate(m: Measure[T], f: Callable[[T], R]) -> R:
    if base_measure(m) is not m:
        return integrate(base_measure(m), lambda x: density(m)(x) * f(x))
    raise NotImplementedError


@functools.singledispatch
def sample(p: Measure[T], **kwargs) -> T:
    ...


@functools.singledispatch
def base_measure(m: Measure[S]) -> Measure[T]:
    return m


@multipledispatch.dispatch
def product(p: Measure[S], q: Measure[T]) -> Measure[tuple[S, T]]:
    p_base = base_measure(p)
    q_base = base_measure(q)
    return Measure(
        base=p_base,
        density=lambda x: density(p, p_base)(x) * density(q, q_base)(x),
    )


@multipledispatch.dispatch
def mixture(p: Measure[S], q: Measure[T]) -> Measure[S | T]:
    p_base = base_measure(p)
    q_base = base_measure(q)
    return Measure(
        base=p_base,
        density=lambda x: density(p, p_base)(x) + density(q, q_base)(x),
    )


@functools.singledispatch
def pushforward(p: Measure[S], f: Callable[[S], T]) -> Measure[T]:
    return Measure(
        base=p,
        density=lambda x: density(p)(f(x)),
    )


@multipledispatch.dispatch
def ratio(p: Measure[T], q: Measure[T], **kwargs) -> Measure[T]:
    q_base = base_measure(q)
    return Measure(
        base=q_base,
        density=lambda x: density_rel(p, q_base)(x) / density(q, q_base)(x),
    )

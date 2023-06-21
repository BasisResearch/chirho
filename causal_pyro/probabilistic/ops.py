from typing import Callable, Container, Dict, Generic, Hashable, Iterable, List, Literal, Optional, Protocol, Set, Tuple, TypeVar, Union

import functools
import math
import numbers

import multipledispatch
import pyro
import torch

R = torch.Tensor | float | int
S = TypeVar("S")
T = TypeVar("T")


###########################################################################
# Types, constructors and accessors
###########################################################################

class Measure(Protocol[T]):
    base_measure: Optional["Measure[T]"]
    log_density: Callable[[T], R]


Kernel = Callable[[S], Measure[T]]


class NewMeasure(Generic[T], Measure[T]):
    base_measure: Measure[T]
    log_density: Callable[[T], R]

    def __init__(self, base_measure: Measure[T], log_density: Callable[[T], R]):
        self.base_measure = base_measure
        self.log_density = log_density


class DiracMeasure(Generic[T], Measure[T]):
    base_measure = None
    value: T

    def __init__(self, value: T):
        self.value = value

    def log_density(self, x: T) -> R:
        return 0. if x == self.value else -math.inf


class LebesgueMeasure(Measure[R]):
    base_measure = None
    log_weight: R

    def __init__(self, log_weight: R = 0.):
        self.weight = log_weight

    def log_density(self, x: R) -> R:
        return self.log_weight


class CountingMeasure(Measure[int]):
    base_measure = None
    n: int

    def __init__(self, n: int):
        self.n = n

    def log_density(self, x: int) -> R:
        return 0. if x == self.n else -math.inf


class GaussianMeasure(Measure[R]):
    base_measure: LebesgueMeasure = LebesgueMeasure(log_weight=-0.5 * math.log(2 * math.pi))
    loc: R
    scale: R

    def __init__(self, loc: R, scale: R):
        self.loc = loc
        self.scale = scale

    def log_density(self, x: R) -> R:
        return -torch.log(torch.as_tensor(self.scale)) - (x - self.loc) ** 2 / (2 * self.scale ** 2)


class BernoulliMeasure(Measure[bool]):
    base_measure: CountingMeasure = CountingMeasure(2)
    p: R

    def __init__(self, p: R):
        self.p = p

    def log_density(self, x: bool) -> R:
        return math.log(self.p) if x else math.log(1 - self.p)


@functools.singledispatch
def as_measure(x, **kwargs) -> Measure[T] | Kernel[S, T]:
    raise NotImplementedError


@functools.singledispatch
def base_measure(m: Measure[T]) -> Measure[T]:
    return m.base_measure if m.base_measure is not None else m


@functools.singledispatch
def log_density(m: Measure[T], other: Optional[Measure[T]] = None) -> Callable[[T], R]:
    if other is not None:
        return log_density(importance(m, other))
    return lambda x: log_density(base_measure(m))(x) + m.log_density(x)


###########################################################################
# Importance combinators for weighting
###########################################################################

@multipledispatch.dispatch
def importance(p: Measure[T], q: Measure[T], **kwargs) -> Measure[T]:
    log_densities = []
    while base_measure(q) is not q:
        p = base_measure(p_base)
        q_base, q_logp = base_measure(q_base), log_density(q_base) + q_logp

    return NewMeasure(
        q_base if p_base is q_base else importance(p_base, q_base, **kwargs),
        lambda x: log_density(p)(x) - log_density(q)(x)
    )


@importance.register
def _importance_pyro(p: PyroMeasure, q: PyroMeasure, **kwargs) -> PyroMeasure:
    ...


###########################################################################
# Elimination forms
###########################################################################

@functools.singledispatch
def integrate(m: Measure[T], f: Optional[Callable[[T], R]] = None, **kwargs) -> R:
    if base_measure(m) is m:
        return integrate(base_measure(m), lambda x: torch.exp(log_density(m)(x)) * f(x))
    raise NotImplementedError


###########################################################################
# Probability measures
###########################################################################

@functools.singledispatch
def is_normalized(m: Measure[T]) -> bool:
    return getattr(m, "__is_normalized__", False)


@functools.singledispatch
def normalize(p: Measure[T], **kwargs) -> Measure[T]:
    if is_normalized(p):
        return p
    raise NotImplementedError


###########################################################################
# Other combinators
###########################################################################

@functools.singledispatch
def pushforward(p: Measure[S], f: Callable[[S], T]) -> Measure[T]:
    ...


@multipledispatch.dispatch
def product(p: Measure[S], q: Measure[T]) -> Measure[tuple[S, T]]:
    ...


@multipledispatch.dispatch
def mixture(p: Measure[S], q: Measure[T]) -> Measure[S | T]:
    ...

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
def measure_from(x, **kwargs) -> Measure[T]:
    raise NotImplementedError


@functools.singledispatch
def base_measure_of(m: Measure[T]) -> Measure[T]:
    return m.base_measure if m.base_measure is not None else m


@multipledispatch.dispatch
def log_density_of(m: Measure[T]) -> Callable[[T], R]:
    return m.log_density


@functools.singledispatch
def tfm_of(m: Measure[T]) -> Callable[[S], T]:
    raise NotImplementedError


@functools.singledispatch
def support_of(m: Measure[T]) -> Container[T]:
    raise NotImplementedError


###########################################################################
# Importance combinators for weighting
###########################################################################

class AbsoluteContinuityError(Exception):
    pass


@multipledispatch.dispatch
def log_density_rel(p: Measure[T], q: Measure[T]) -> Callable[[T], R]:
    if base_measure_of(p) is not p:
        return lambda x: log_density_of(p)(x) + log_density_rel(base_measure_of(p), q)(x)
    elif base_measure_of(q) is not q:
        return lambda x: -log_density_of(q)(x) + log_density_rel(p, base_measure_of(q))(x)
    else:
        raise AbsoluteContinuityError


@multipledispatch.dispatch
def importance(p: Measure[T], q: Measure[T]) -> Measure[T]:
    return NewMeasure(q, log_density_rel(p, q))


@functools.singledispatch
def log(x):
    raise NotImplementedError


@functools.singledispatch
def exp(x):
    raise NotImplementedError


@multipledispatch.dispatch
def mul(x, y):
    raise NotImplementedError


###########################################################################
# Elimination forms
###########################################################################

@functools.singledispatch
def integrate(m: Measure[T], f: Callable[[T], R]) -> R:
    if base_measure_of(m) is not m:
        return integrate(base_measure_of(m), lambda x: math.exp(log_density_of(m)(x)) * f(x))
    raise NotImplementedError


###########################################################################
# Probability measures
###########################################################################

@functools.singledispatch
def is_normalized(m: Measure[T]) -> bool:
    return getattr(m, "__is_normalized__", False)


@functools.singledispatch
def normalize(p: Measure[T]) -> Measure[T]:
    if is_normalized(p):
        return p
    raise NotImplementedError


def expectation(p: Measure[T], f: Callable[[T], R]) -> R:
    return integrate(normalize(p), f)


def kl(p: Measure[T], q: Measure[T]) -> R:
    p, q = normalize(p), normalize(q)
    return integrate(p, log_density_of(importance(p, q)))


def elbo(p: Measure[T], q: Measure[T]) -> R:
    q = normalize(q)
    return integrate(q, log_density_of(importance(p, q)))


###########################################################################
# Other measure combinators
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

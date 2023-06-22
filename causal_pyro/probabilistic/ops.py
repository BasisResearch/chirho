from typing import Callable, Generic, Optional, TypeVar

import functools
import math

import multipledispatch
import torch

R = torch.Tensor | float | int
S = TypeVar("S")
T = TypeVar("T")


###########################################################################
# Types, constructors and accessors
###########################################################################

class Measure(Generic[T]):

    def log_density(self, x: T) -> R:
        raise NotImplementedError

    @property
    def base_measure(self) -> "Measure[T]":
        return self


@functools.singledispatch
def as_measure(m, **kwargs) -> Measure:
    raise NotImplementedError


@as_measure.register
def _as_measure_measure(
    m: Measure[T], *, log_density: Optional[Callable[[T], R]] = None,
) -> Measure[T]:
    return m if log_density is None else _NewMeasure(m, log_density)


class _NewMeasure(Generic[T], Measure[T]):
    def __init__(self, base_measure: Measure[T], log_density: Callable[[T], R]):
        self._base_measure = base_measure
        self._log_density = log_density

    @property
    def base_measure(self) -> Measure[T]:
        return self._base_measure

    def log_density(self, x: T) -> R:
        return self._log_density(x)


class DiracMeasure(Generic[T], Measure[T]):
    value: T

    def __init__(self, value: T):
        self.value = value

    def log_density(self, x: T) -> R:
        return 0. if x == self.value else -math.inf


class LebesgueMeasure(Measure[R]):
    log_weight: R
    d: int

    def __init__(self, log_weight: R = 0., d: int = 1):
        self.log_weight = log_weight
        self.d = d

    def log_density(self, x) -> R:
        return self.log_weight


class CountingMeasure(Measure[int]):
    n: int

    def __init__(self, n: int):
        self.n = n

    def log_density(self, x: int) -> R:
        return -math.log(self.n)


class GaussianMeasure(Measure[R]):
    loc: R
    scale: R

    def __init__(self, loc: R, scale: R):
        self.loc = loc
        self.scale = scale

    @property
    def base_measure(self) -> LebesgueMeasure:
        return LebesgueMeasure(log_weight=-0.5 * math.log(2 * math.pi), d=1)

    def log_density(self, x: R) -> R:
        return -math.log(self.scale) - (x - self.loc) ** 2 / (2 * self.scale ** 2)


class BernoulliMeasure(Measure[bool]):
    p: R

    def __init__(self, p: R):
        self.p = p

    @property
    def base_measure(self) -> Measure[int]:
        return CountingMeasure(2)

    def log_density(self, x: bool) -> R:
        return math.log(self.p) if x else math.log(1 - self.p)


###########################################################################
# Importance combinators for weighting
###########################################################################

class AbsoluteContinuityError(Exception):
    pass


@multipledispatch.dispatch(Measure, Measure)
def importance(p: Measure[T], q: Measure[T]) -> Measure[T]:
    if p.base_measure is not p:
        dp, ddq = p.log_density, importance(p.base_measure, q).log_density
        return as_measure(q, log_density=lambda x: ddq(x) + dp(x))
    elif q.base_measure is not q:
        dq, ddp = q.log_density, importance(p, q.base_measure).log_density
        return as_measure(q, log_density=lambda x: -dq(x) + ddp(x))
    else:
        raise AbsoluteContinuityError


@importance.register(LebesgueMeasure, LebesgueMeasure)
def _importance_lebesgue(p: LebesgueMeasure, q: LebesgueMeasure) -> LebesgueMeasure:
    if p.d != q.d:
        raise AbsoluteContinuityError
    return LebesgueMeasure(log_weight=p.log_weight - q.log_weight)


@importance.register(CountingMeasure, CountingMeasure)
def _importance_counting(p: CountingMeasure, q: CountingMeasure) -> CountingMeasure:
    if p.n != q.n:
        raise AbsoluteContinuityError
    return CountingMeasure(p.n)


###########################################################################
# Integration
###########################################################################

@functools.singledispatch
def integrate(m: Measure[T], f: Optional[Callable[[T], R]] = None) -> R:
    raise NotImplementedError


###########################################################################
# Probability measures
###########################################################################

@functools.singledispatch
def normalize(p: Measure[T]) -> Measure[T]:
    raise NotImplementedError


def expectation(p: Measure[T], f: Callable[[T], R]) -> R:
    return integrate(normalize(p), f)

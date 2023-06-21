from typing import Callable, Generic, Optional, Protocol, TypeVar

import functools
import math
import typing

import multipledispatch
import torch

R = torch.Tensor | float | int
S = TypeVar("S")
T = TypeVar("T")


###########################################################################
# Types, constructors and accessors
###########################################################################

@typing.runtime_checkable
class Measure(Protocol[T]):

    def log_density(self, x: T) -> R: ...

    @property
    def base_measure(self) -> "Measure[T]": ...


@functools.singledispatch
def measure_from(x, **kwargs) -> Measure[T]:
    raise NotImplementedError


@functools.singledispatch
def log_density_of(m: Measure[T]) -> Callable[[T], R]:
    raise NotImplementedError


@functools.singledispatch
def base_measure_of(m: Measure[T]) -> Measure[T]:
    raise NotImplementedError


class AbstractMeasure(Generic[T], Measure[T]):

    def log_density(self, x: T) -> R:
        raise NotImplementedError

    @property
    def base_measure(self) -> Measure[T]:
        return self


@log_density_of.register(AbstractMeasure)
def _log_density_of_abstract(m: AbstractMeasure[T]) -> Callable[[T], R]:
    return m.log_density


@base_measure_of.register(AbstractMeasure)
def _base_measure_of_abstract(m: AbstractMeasure[T]) -> Measure[T]:
    return m.base_measure


class NewMeasure(Generic[T], AbstractMeasure[T]):
    def __init__(self, base_measure: Measure[T], log_density: Callable[[T], R]):
        self._base_measure = base_measure
        self._log_density = log_density

    @property
    def base_measure(self) -> Measure[T]:
        return self._base_measure
    
    def log_density(self, x: T) -> R:
        return self._log_density(x)


class DiracMeasure(Generic[T], AbstractMeasure[T]):
    value: T

    def __init__(self, value: T):
        self.value = value

    def log_density(self, x: T) -> R:
        return 0. if x == self.value else -math.inf


class LebesgueMeasure(AbstractMeasure[R]):
    log_weight: R
    d: int

    def __init__(self, log_weight: R = 0., d: int = 1):
        self.log_weight = log_weight
        self.d = d

    def log_density(self, x) -> R:
        return self.log_weight


class CountingMeasure(AbstractMeasure[int]):
    n: int

    def __init__(self, n: int):
        self.n = n

    def log_density(self, x: int) -> R:
        return -math.log(self.n)


class GaussianMeasure(AbstractMeasure[R]):
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


class BernoulliMeasure(AbstractMeasure[bool]):
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


@multipledispatch.dispatch(AbstractMeasure, AbstractMeasure)
def importance(p: Measure[T], q: Measure[T]) -> Measure[T]:
    if base_measure_of(p) is not p:
        dp, ddq = log_density_of(p), log_density_of(importance(base_measure_of(p), q))
        return NewMeasure(q, lambda x: ddq(x) + dp(x))
    elif base_measure_of(q) is not q:
        dq, ddp = log_density_of(q), log_density_of(importance(p, base_measure_of(q)))
        return NewMeasure(q, lambda x: -dq(x) + ddp(x))
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
def is_normalized(m: Measure[T]) -> bool:
    return getattr(m, "__normalized__", False)


@functools.singledispatch
def normalize(p: Measure[T]) -> Measure[T]:
    if is_normalized(p):
        return p
    raise NotImplementedError


def expectation(p: Measure[T], f: Callable[[T], R]) -> R:
    return integrate(normalize(p), f)

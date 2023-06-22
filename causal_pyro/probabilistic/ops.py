from typing import Callable, Generic, Optional, TypeVar

import functools

import multipledispatch
import torch

from .defer_args import defer_args

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


###########################################################################
# Importance combinators for weighting
###########################################################################

class AbsoluteContinuityError(Exception):
    pass


@defer_args(Measure)
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


###########################################################################
# Integration
###########################################################################

@defer_args(Measure)
@functools.singledispatch
def integrate(m: Measure[T], f: Optional[Callable[[T], R]] = None) -> R:
    raise NotImplementedError


@defer_args(Measure)
@functools.singledispatch
def normalize(p: Measure[T]) -> Measure[T]:
    raise NotImplementedError

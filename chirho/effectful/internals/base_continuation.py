from typing import Callable, Generic, Optional, ParamSpec, TypeVar

from chirho.effectful.ops.continuation import Continuation
from chirho.effectful.ops.operation import define
from chirho.effectful.ops.interpretation import register

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


class AffineContinuationError(Exception):
    pass


@register(define(Continuation))
class _BaseAffineContinuation(Generic[S, T]):
    cont: Callable[[Optional[T], Optional[S]], T]
    used: bool

    def __init__(self, __cont: Callable[[Optional[T], Optional[S]], T]):
        self.cont = __cont
        self.used = False

    def __call__(self, __res: Optional[T], __value: Optional[S]) -> T:
        try:
            if self.used:
                raise AffineContinuationError(f"can use {self.cont} at most once")
            return self.cont(__res, __value)
        finally:
            self.used = True

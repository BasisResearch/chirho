from typing import Callable, Generic, Optional, ParamSpec, TypeVar

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


class _BaseAffineContinuation(Generic[S, T]):
    cont: Callable[[Optional[T], Optional[S]], T]
    used: bool

    def __init__(self, cont: Callable[[Optional[T], Optional[S]], T]):
        self.cont = cont
        self.used = False

    def __call__(self, _res: Optional[T], value: Optional[S]) -> T:
        try:
            if self.used:
                from chirho.effectful.ops.continuation import AffineContinuationError
                raise AffineContinuationError(f"can use {self.cont} at most once")
            return self.cont(_res, value)
        finally:
            self.used = True

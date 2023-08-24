from typing import Callable, Generic, Optional, ParamSpec, TypeVar

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


class _BaseAffineContinuation(Generic[T]):
    cont: Callable[[Optional[T], Optional[T]], T]
    used: bool

    def __init__(self, cont: Callable[[Optional[T], Optional[T]], T]):
        self.cont = cont
        self.used = False

    def __call__(self, _res: Optional[T], value: Optional[T]) -> T:
        try:
            if self.used:
                from chirho.effectful.ops.continuation import AffineContinuationError
                raise AffineContinuationError(f"can use {self.cont} at most once")
            return self.cont(_res, value)
        finally:
            self.used = True

from typing import Callable, Generic, Optional, TypeVar

import functools

from chirho.effectful.ops.interpretation import Interpretation, interpreter
from chirho.effectful.ops.operation import Operation, define

S = TypeVar("S")
T = TypeVar("T")


Continuation = Callable[..., T]


class AffineContinuationError(Exception):
    pass


class _AffineContinuation(Generic[T]):
    cont: Continuation[T]
    used: bool

    def __init__(self, cont: Continuation[T]):
        self.cont = cont
        self.used = False

    def __call__(self, *args, **kwargs) -> T:
        try:
            if self.used:
                raise AffineContinuationError(f"can use {self.cont} at most once")
            return self.cont(*args, **kwargs)
        finally:
            self.used = True


@define(Operation)
def push_prompts(conts: Interpretation[T]) -> Callable[[Callable[..., T]], Callable[..., T]]:

    from ..internals.runtime import get_interpretation

    # TODO switch to using contextlib.contextmanager here
    def _decorator(fn: Callable[..., T]) -> Callable[..., T]:

        @functools.wraps(fn)
        def _wrapper(result: Optional[T], *args: T, **kwargs) -> T:

            # TODO handle argument capture separately and just use conts here
            conts_ = define(Interpretation)({
                p: functools.partial(
                    lambda k, a, kw, _, res: k(res, *a, **kw),
                    conts[p], args, kwargs
                ) for p in conts.keys()
            })

            resets = define(Interpretation)({
                p: _AffineContinuation(
                    interpreter(define(Interpretation)({
                        p: get_interpretation()[p] if p in get_interpretation() else p.default
                    }))(conts_[p])
                )
                for p in conts.keys()
            })

            return interpreter(resets)(fn)(result, *args, **kwargs)

        return _wrapper

    return _decorator

from typing import Callable, Generic, Optional, TypeVar

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
def reset_prompt(
    prompt_op: Operation[T],
    continuation: Continuation[T],
    fn: Callable[..., T],
    result: Optional[T],
    *args: T,
    **kwargs,
) -> T:
    from ..internals.runtime import get_interpretation

    if prompt_op in get_interpretation():
        prev_continuation = get_interpretation()[prompt_op]
    else:
        prev_continuation = prompt_op.default

    continuation = _AffineContinuation(continuation)
    prev_continuation: Continuation[T] = _AffineContinuation(prev_continuation)

    shift = define(Interpretation)({prompt_op: prev_continuation})
    reset = define(Interpretation)(
        {
            prompt_op: lambda _, res: interpreter(shift)(continuation)(
                res, *args, **kwargs
            )
        }
    )
    return interpreter(reset)(fn)(result, *args, **kwargs)
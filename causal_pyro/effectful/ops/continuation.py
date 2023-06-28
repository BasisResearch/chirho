from typing import Callable, Generic, Optional, TypeVar

from causal_pyro.effectful.ops.interpretation import Interpretation, interpreter
from causal_pyro.effectful.ops.operation import Operation, define


S = TypeVar("S")
T = TypeVar("T")


Continuation = Callable[..., T]


class _AffineContinuation(Generic[T]):
    cont: Continuation[T]
    used: bool

    def __init__(self, cont: Continuation[T]):
        self.cont = cont
        self.used = False

    def __call__(self, *args, **kwargs) -> T:
        try:
            if not self.used:
                return self.cont(*args, **kwargs)
            else:
                raise ValueError(f"can use continuation {self.cont} at most once")
        finally:
            self.used = True


@define(Operation)
def reset_prompt(
    prompt_op: Operation[T],
    continuation: Continuation[T],
    fn: Callable[..., T],
    result: Optional[T],
    *args: T,
    **kwargs
) -> T:
    from ..internals.runtime import get_interpretation

    if prompt_op in get_interpretation():
        prev_continuation = get_interpretation()[prompt_op]
    else:
        prev_continuation = prompt_op.default

    continuation = _AffineContinuation(continuation)
    prev_continuation = _AffineContinuation(prev_continuation)

    shift = define(Interpretation)({prompt_op: prev_continuation})
    reset = define(Interpretation)({
        prompt_op: lambda _, res: interpreter(shift)(continuation)(res, *args, **kwargs)
    })
    return interpreter(reset)(fn)(result, *args, **kwargs)


def prompt_calls(prompt_op: Operation[T], *ops: Operation[T]) -> Interpretation[T]:
    return define(Interpretation)({
        op: lambda res, *args, **kwargs: prompt_op(res)
        for op in set(ops)
    })

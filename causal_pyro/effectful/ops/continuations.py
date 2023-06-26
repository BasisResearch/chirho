from typing import Callable, Generic, Optional, TypeVar

from causal_pyro.effectful.ops.operations import Interpretation, Operation, \
    define, interpreter


S = TypeVar("S")
T = TypeVar("T")


def prompt_calls(prompt_op: Operation[T], *ops: Operation[T]) -> Interpretation[T]:
    return define(Interpretation)({
        op: lambda res, *args, **kwargs: prompt_op(res)
        for op in set(ops)
    })


class AffineContinuation(Generic[T]):
    cont: Callable[..., T]
    used: bool

    def __init__(self, cont: Callable[..., T]):
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
    cont: Callable[..., T],
    fst: Callable[..., T],
    result: Optional[T],
    *args: T,
    **kwargs
) -> T:
    from ._runtime import get_interpretation

    if prompt_op in get_interpretation():
        prev_cont = get_interpretation()[prompt_op]
    else:
        prev_cont = prompt_op.default

    cont = AffineContinuation(cont)
    prev_cont = AffineContinuation(prev_cont)

    shift = define(Interpretation)({prompt_op: prev_cont})
    reset = define(Interpretation)({
        prompt_op: lambda _, res: interpreter(shift)(cont)(res, *args, **kwargs)
    })
    return interpreter(reset)(fst)(result, *args, **kwargs)

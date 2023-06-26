from typing import Callable, Optional, TypeVar

from causal_pyro.effectful.ops.operations import Interpretation, Operation, \
    define, interpreter


S = TypeVar("S")
T = TypeVar("T")


def prompt_calls(prompt_op: Operation[T], *ops: Operation[T]) -> Interpretation[T]:
    return define(Interpretation)({
        op: lambda res, *args, **kwargs: prompt_op(res)
        for op in set(ops)
    })


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

    shift = define(Interpretation)({prompt_op: prev_cont})
    reset = define(Interpretation)({
        prompt_op: lambda _, res: interpreter(shift)(cont)(res, *args, **kwargs)
    })
    return interpreter(reset)(fst)(result, *args, **kwargs)

from typing import Callable, ClassVar, Generic, Iterable, Optional, TypedDict, TypeVar

import contextlib
import functools

from causal_pyro.effectful.ops.operations import Interpretation, Operation, StatefulInterpretation, \
    define, interpreter, register


S = TypeVar("S")
T = TypeVar("T")


def prompt_calls(prompt_op: Operation[T], *ops: Operation[T]) -> Interpretation[T]:
    return define(Interpretation)({
        op: lambda res, *args, **kwargs: prompt_op(res)
        for op in set(ops)
    })


# @define(Operation)
def shift_prompt(prompt_op: Operation[T], cont: Callable[..., T], fst: Callable[..., T]) -> Callable[..., T]:

    def _wrapped_fst(res, *args, **kwargs):
        # fst_ = handler({prompt_op: lambda _, res: cont_(res, *args, **kwargs)})(fst)
        cont_ = cont
        fst_ = interpreter({prompt_op: lambda _, res: cont_(res, *args, **kwargs)})(fst)
        # fst_ = handler({prompt_op: lambda _, res: cont(res, *args, **kwargs)})(fst)
        return fst_(res, *args, **kwargs)
    
    return _wrapped_fst
    # return functools.partial(reset, cont, fst)


@define(Operation)
def shift(result: Optional[T]) -> T:
    return result


@define(Operation)
def reset(
    continuation: Callable[..., T],
    fn: Callable[..., T],
    result: Optional[T],
    *active_args: T,
    **active_kwargs
) -> Callable[..., T]:
    ...


class ContState(TypedDict):
    continuation: Optional[Callable]  # [..., T]
    result: Optional[T]  # [T]
    args: list  # [T]
    kwargs: dict  # [str, T]
    # active_op: Operation  # [T]


class OneShotContinuationInterpretation(Generic[T], StatefulInterpretation[ContState, T]):
    state: ContState


@register(reset, OneShotContinuationInterpretation)
def _interpret_reset(
    state: ContState,
    _res: Optional[T],
    continuation: Callable[..., T],
    fn: Callable[..., T],
    result: Optional[T],
    *args: T,
    **kwargs
) -> Callable[..., T]:

    state["continuation"] = continuation
    state["result"] = result
    state["args"] = list(*args)
    state["kwargs"] = dict(**kwargs)
    return fn(result, *args, **kwargs)


@register(shift, OneShotContinuationInterpretation)
def _interpret_shift(
    state: ContState,
    _res: Optional[T],
    new_result: Optional[T],
    # *new_args: Optional[T],
    # **new_kwargs
) -> T:
    if state["continuation"] is None:
        raise RuntimeError("shift used outside of reset, or used multiple times in the same reset")
    else:
        # require that the continuation is called at most once
        continuation, state["continuation"] = state["continuation"], None

    state["result"] = new_result if new_result is not None else state["result"]
    # state["args"] = [
    #     new_arg if new_arg is not None else arg
    #     for arg, new_arg in zip(state["args"], new_args)
    # ]
    # state["kwargs"].update(new_kwargs)

    return continuation(state["result"], *state["args"], **state["kwargs"])

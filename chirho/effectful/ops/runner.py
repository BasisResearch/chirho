import contextlib
from typing import Optional, ParamSpec, TypeVar

from chirho.effectful.ops.handler import fwd
from chirho.effectful.ops.interpretation import Interpretation, bind_and_push_prompts, bind_result, interpreter
from chirho.effectful.ops.operation import Operation, define

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")

Prompt = Operation[[Optional[T]], T]


@define(Operation)
def reflect(__result: Optional[T]) -> T:
    return __result


@define(Operation)
def product(
    intp: Interpretation[S, T],
    *intps: Interpretation[S, T],
    prompt: Prompt[T] = reflect,
) -> Interpretation[S, T]:
    if len(intps) == 0:  # unit
        return intp
    elif len(intps) > 1:  # associativity
        return product(intp, product(*intps, prompt=prompt), prompt=prompt)

    (intp2,) = intps
    reflect_intp_ops = {
        op: bind_result(lambda v, *_, **__: prompt(v))
        for op in set(intp.keys()) - set(intp2.keys())
    }

    # on prompt, jump to the outer interpretation and interpret it using itself
    return {
        op: bind_and_push_prompts({prompt: interpreter(intp)(op)})(
            interpreter(reflect_intp_ops)(interpreter(intp2)(intp2[op]))
        ) for op in intp2.keys()
    }


def handler_prompt_to_runner_prompt(
    intp: Interpretation[S, T], handler_prompt: Prompt[T], runner_prompt: Prompt[T],
) -> Interpretation[S, T]:
    return {  # TODO shallow_interpreter?
        op: interpreter({handler_prompt: runner_prompt})(intp[op])
        for op in intp.keys()
    }


@contextlib.contextmanager
def runner(
    intp: Interpretation[S, T],
    *,
    prompt: Prompt[T] = reflect,
    handler_prompt: Optional[Prompt[T]] = None,
):
    from .runtime import get_interpretation

    curr_intp, next_intp = get_interpretation(), intp

    if handler_prompt is not None:
        assert prompt is not handler_prompt, \
            f"runner prompt and handler prompt must be distinct, but got {handler_prompt}"
        curr_intp = handler_prompt_to_runner_prompt(curr_intp, handler_prompt, prompt)
        next_intp = handler_prompt_to_runner_prompt(next_intp, handler_prompt, prompt)

    with interpreter(product(curr_intp, next_intp, prompt=prompt)):
        yield intp

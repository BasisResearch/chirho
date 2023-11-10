import contextlib
from typing import Optional, ParamSpec, TypeVar

from chirho.effectful.ops.handler import fwd
from chirho.effectful.ops.interpretation import Interpretation, bind_and_push_prompts, bind_result, interpreter, shallow_interpreter
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

    # cases:
    # 1. op in intp2 but not intp: handle from scratch when encountered in latent context
    # 2. op in intp but not intp2: don't expose in user code
    # 3. op in both intp and intp2: use intp[op] under intp and intp2[op] under intp2 as continuations
    (intp2,) = intps

    reflect_intp_ops = {
        op: bind_result(lambda v, *_, **__: prompt(v))
        for op in set(intp.keys()) - set(intp2.keys())
    }

    # on reflect, jump to the outer interpretation and interpret it using itself
    return {
        op: bind_and_push_prompts({prompt: interpreter(intp)(op)})(
            interpreter(reflect_intp_ops)(interpreter(intp2)(intp2[op]))
        ) for op in intp2.keys()
    }


def handler_prompt_to_runner_prompt(
    intp: Interpretation[S, T], handler_prompt: Prompt[T], runner_prompt: Prompt[T],
) -> Interpretation[S, T]:
    return {
        op: shallow_interpreter({handler_prompt: runner_prompt})(intp[op])
        for op in intp.keys()
    }


@contextlib.contextmanager
def runner(
    intp: Interpretation[S, T],
    *,
    prompt: Prompt[T] = reflect,
    handler_prompt: Prompt[T] = fwd,
):
    from .runtime import get_interpretation

    assert prompt is not handler_prompt, "runner prompt and handler prompt must be distinct"
    curr_intp = handler_prompt_to_runner_prompt(get_interpretation(), handler_prompt, prompt)
    next_intp = handler_prompt_to_runner_prompt(intp, handler_prompt, prompt)

    with interpreter(product(curr_intp, next_intp, prompt=prompt)):
        yield intp

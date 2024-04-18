import contextlib
from typing import Mapping, Optional, ParamSpec, TypeVar

from chirho.meta.ops.interpreter import (
    Prompt,
    bind_prompts,
    bind_result,
    interpreter,
)
from chirho.meta.ops.core import Interpretation, Operation, define

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


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
        op: bind_prompts({prompt: interpreter(intp)(op)})(
            interpreter(reflect_intp_ops)(interpreter(intp2)(intp2[op]))
        )
        for op in intp2.keys()
    }


@contextlib.contextmanager
def runner(
    intp: Interpretation[S, T],
    *,
    prompt: Prompt[T] = reflect,
    handler_prompt: Optional[Prompt[T]] = None,
):
    from ..internals.runtime import get_interpretation

    curr_intp, next_intp = get_interpretation(), intp

    if handler_prompt is not None:
        assert (
            prompt is not handler_prompt
        ), f"runner prompt and handler prompt must be distinct, but got {handler_prompt}"
        h2r = {handler_prompt: prompt}
        curr_intp = {op: interpreter(h2r)(curr_intp[op]) for op in curr_intp.keys()}
        next_intp = {op: interpreter(h2r)(next_intp[op]) for op in next_intp.keys()}

    with interpreter(product(curr_intp, next_intp, prompt=prompt)):
        yield intp

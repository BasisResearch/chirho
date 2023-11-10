import contextlib
from typing import Optional, ParamSpec, TypeVar

from chirho.effectful.ops.interpretation import Interpretation, bind_and_push_prompts, interpreter
from chirho.effectful.ops.operation import Operation, define

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")

Prompt = Operation[[Optional[T]], T]


@define(Operation)
def fwd(__result: Optional[T]) -> T:
    return __result


@define(Operation)
def compose(
    intp: Interpretation[S, T],
    *intps: Interpretation[S, T],
    prompt: Prompt[T] = fwd,
) -> Interpretation[S, T]:
    if len(intps) == 0:  # unit
        return intp
    elif len(intps) > 1:  # associativity
        return compose(intp, compose(*intps, prompt=prompt), prompt=prompt)

    (intp2,) = intps
    return dict(
        [(op, intp[op]) for op in set(intp.keys()) - set(intp2.keys())]
        + [(op, intp2[op]) for op in set(intp2.keys()) - set(intp.keys())]
        + [
            (op, bind_and_push_prompts({prompt: intp[op]})(intp2[op]))
            for op in set(intp.keys()) & set(intp2.keys())
        ]
    )


@contextlib.contextmanager
def handler(intp: Interpretation[S, T], *, prompt: Prompt[T] = fwd):
    from .runtime import get_interpretation

    with interpreter(compose(get_interpretation(), intp, prompt=prompt)):
        yield intp

import contextlib
import functools
import typing
from typing import Optional, ParamSpec, TypeVar

from chirho.effectful.ops.continuation import bind_and_push_prompts, shallow_interpreter
from chirho.effectful.ops.handler import fwd
from chirho.effectful.ops.interpretation import Interpretation, interpreter
from chirho.effectful.ops.operation import Operation, define

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
    reflect: Operation[[Optional[T]], T] = reflect,
    fwd: Operation[[Optional[T]], T] = fwd,
) -> Interpretation[S, T]:
    if len(intps) == 0:
        return intp  # unit
    elif len(intps) > 1:  # associativity
        return product(
            intp, product(*intps, reflect=reflect, fwd=fwd), reflect=reflect, fwd=fwd
        )

    # cases:
    # 1. op in intp2 but not intp: handle from scratch when encountered in latent context
    # 2. op in intp but not intp2: don't expose in user code
    # 3. op in both intp and intp2: use intp[op] under intp and intp2[op] under intp2 as continuations
    (intp2,) = intps

    block_outer = define(Interpretation)({
        op: shallow_interpreter({fwd: lambda _, v: reflect(v)})(intp[op])
        for op in intp.keys()
    })

    block_inner = define(Interpretation)({
        op: shallow_interpreter({fwd: lambda _, v: reflect(v)})(intp2[op])
        for op in set(intp2.keys()) & set(intp.keys())
    })

    # on reflect, jump to the outer interpretation and interpret it using itself
    return define(Interpretation)({
        op: bind_and_push_prompts(
            {reflect: interpreter(block_outer)(define(Operation)(op).default)},
        )(interpreter(block_inner)(intp2[op]))
        for op in intp2.keys()
    })


@define(Operation)
@contextlib.contextmanager
def runner(
    intp: Interpretation[S, T],
    *,
    reflect: Operation[[Optional[T]], T] = reflect,
    fwd: Operation[[Optional[T]], T] = fwd,
):
    from .runtime import get_interpretation

    with interpreter(product(get_interpretation(), intp, reflect=reflect, fwd=fwd)):
        yield intp

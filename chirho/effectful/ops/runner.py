import contextlib
import functools
from typing import Optional, ParamSpec, TypeVar

from chirho.effectful.ops.continuation import bind_and_push_prompts
from chirho.effectful.ops.handler import fwd
from chirho.effectful.ops.interpretation import Interpretation, interpreter
from chirho.effectful.ops.operation import Operation, define

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@define(Operation)
def reflect(result: Optional[T]) -> T:
    return result


@define(Operation)
def product(
    intp: Interpretation[T],
    *intps: Interpretation[T],
    reflect: Operation[[Optional[T]], T] = reflect,
    fwd: Operation[[Optional[T]], T] = fwd,
) -> Interpretation[T]:
    if len(intps) == 0:
        return intp  # unit
    elif len(intps) > 1:  # associativity
        return product(
            intp, product(*intps, reflect=reflect, fwd=fwd), reflect=reflect, fwd=fwd
        )

    def _op_or_result(
        op: Operation[P, T], result: Optional[T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        return result if result is not None else op(*args, **kwargs)

    # cases:
    # 1. op in intp2 but not intp: handle from scratch when encountered in latent context
    # 2. op in intp but not intp2: don't expose in user code
    # 3. op in both intp and intp2: use intp[op] under intp and intp2[op] under intp2 as continuations
    (intp2,) = intps

    intp_outer = define(Interpretation)(
        {
            op: bind_and_push_prompts({fwd: lambda _, v: reflect(v)}, op, intp[op])
            for op in intp.keys()
        }
    )

    intp_inner = define(Interpretation)(
        {
            op: bind_and_push_prompts({fwd: lambda _, v: reflect(v)}, op, intp2[op])
            for op in intp2.keys()
            if op in intp
        }
    )

    # on reflect, jump to the outer interpretation and interpret it using itself
    return define(Interpretation)(
        {
            op: bind_and_push_prompts(
                {
                    reflect: interpreter(intp_outer)(
                        functools.partial(_op_or_result, op)
                    )
                },
                op,
                interpreter(intp_inner)(intp2[op]),
            )
            for op in intp2.keys()
        }
    )


@define(Operation)
@contextlib.contextmanager
def runner(
    intp: Interpretation[T],
    *,
    reflect: Operation[[Optional[T]], T] = reflect,
    fwd: Operation[[Optional[T]], T] = fwd,
):
    from ..internals.runtime import get_interpretation

    with interpreter(product(get_interpretation(), intp, reflect=reflect, fwd=fwd)):
        yield intp

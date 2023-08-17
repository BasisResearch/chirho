import contextlib
from typing import Optional, TypeVar

from chirho.effectful.ops.continuation import bind_and_push_prompts
from chirho.effectful.ops.interpretation import Interpretation, interpreter
from chirho.effectful.ops.operation import Operation, define

S = TypeVar("S")
T = TypeVar("T")


@define(Operation)
def fwd(result: Optional[T]) -> T:
    return result


@define(Operation)
def compose(
    intp: Interpretation[T], *intps: Interpretation[T], fwd: Operation[T] = fwd
) -> Interpretation[T]:
    if len(intps) == 0:
        return intp  # unit
    elif len(intps) > 1:
        return compose(intp, compose(*intps, fwd=fwd), fwd=fwd)  # associativity

    (intp2,) = intps
    return define(Interpretation)(
        [(op, intp[op]) for op in set(intp.keys()) - set(intp2.keys())]
        + [(op, intp2[op]) for op in set(intp2.keys()) - set(intp.keys())]
        + [
            (op, bind_and_push_prompts({fwd: intp[op]}, op, intp2[op]))
            for op in set(intp.keys()) & set(intp2.keys())
        ]
    )


@define(Operation)
@contextlib.contextmanager
def handler(intp: Interpretation[T], *, fwd: Operation[T] = fwd):
    from ..internals.runtime import get_interpretation

    with interpreter(compose(get_interpretation(), intp, fwd=fwd)):
        yield intp

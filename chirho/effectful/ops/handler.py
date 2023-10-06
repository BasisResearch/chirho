import contextlib
import functools
from typing import Callable, Concatenate, Mapping, Optional, ParamSpec, TypeVar

from chirho.effectful.ops.interpretation import Interpretation, interpreter, shallow_interpreter
from chirho.effectful.ops.operation import Operation, define

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


def bind_and_push_prompts(
    unbound_conts: Interpretation[S, T],
) -> Callable[[Callable[Concatenate[Optional[T], P], T]], Callable[Concatenate[Optional[T], P], T]]:

    @define(Operation)
    def get_args() -> tuple[tuple, dict]:
        raise ValueError("No args stored")

    def _capture_args(
        fn: Callable[Concatenate[Optional[T], Q], T]
    ) -> Callable[Concatenate[Optional[T], Q], T]:

        @functools.wraps(fn)
        def _wrapper(__res: Optional[T], *a: Q.args, **ks: Q.kwargs) -> T:
            return interpreter({get_args: lambda _: (a, ks)})(fn)(__res, *a, **ks)

        return _wrapper

    def _bind_args(
        unbound_conts: Mapping[Operation[[Optional[S]], S], Callable[Concatenate[Optional[T], Q], T]],
    ) -> Interpretation[S, T]:
        return {
            p: functools.partial(
                lambda k, _, res: k(res, *get_args()[0], **get_args()[1]),
                unbound_conts[p],
            ) for p in unbound_conts.keys()
        }

    def _decorator(fn: Callable[Concatenate[Optional[T], P], T]):
        return shallow_interpreter(_bind_args(unbound_conts))(_capture_args(fn))

    return _decorator


@define(Operation)
def fwd(__result: Optional[T]) -> T:
    return __result


@define(Operation)
def compose(
    intp: Interpretation[S, T],
    *intps: Interpretation[S, T],
    fwd: Operation[[Optional[T]], T] = fwd
) -> Interpretation[S, T]:
    if len(intps) == 0:
        return intp  # unit
    elif len(intps) > 1:
        return compose(intp, compose(*intps, fwd=fwd), fwd=fwd)  # associativity

    (intp2,) = intps
    return dict(
        [(op, intp[op]) for op in set(intp.keys()) - set(intp2.keys())]
        + [(op, intp2[op]) for op in set(intp2.keys()) - set(intp.keys())]
        + [
            (op, bind_and_push_prompts({fwd: intp[op]})(intp2[op]))
            for op in set(intp.keys()) & set(intp2.keys())
        ]
    )


@contextlib.contextmanager
def handler(intp: Interpretation[S, T], *, fwd: Operation[[Optional[T]], T] = fwd):
    from .runtime import get_interpretation

    with interpreter(compose(get_interpretation(), intp, fwd=fwd)):
        yield intp

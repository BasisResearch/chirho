import contextlib
import functools
from typing import Callable, Mapping, Optional, ParamSpec, TypeVar

from chirho.effectful.ops.interpretation import Interpretation, get_result, interpreter, shallow_interpreter
from chirho.effectful.ops.operation import Operation, define

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


LocalState = tuple[tuple, dict]


def bind_and_push_prompts(
    unbound_conts: Mapping[Operation[[Optional[S]], S], Callable[P, T]],
) -> Callable[[Callable[P, T]], Callable[P, T]]:

    @define(Operation)
    def get_local_state() -> LocalState:
        raise ValueError("No args stored")

    def _init_local_state(fn: Callable[Q, V]) -> Callable[Q, V]:

        @functools.wraps(fn)
        def _wrapper(*a: Q.args, **ks: Q.kwargs) -> V:
            return interpreter({get_local_state: lambda: (a, ks)})(fn)(*a, **ks)

        return _wrapper

    def _bind_local_state(
        unbound_conts: Mapping[Operation[[Optional[V]], V], Callable[Q, T]],
    ) -> Mapping[Operation[[Optional[V]], V], Callable[[Optional[T]], T]]:
        return {
            p: _set_result_state(functools.partial(
                lambda k, _: k(*get_local_state()[0], **get_local_state()[1]),
                unbound_conts[p],
            )) for p in unbound_conts.keys()
        }

    def _set_result_state(fn: Callable[[Optional[V]], V]) -> Callable[[Optional[V]], V]:

        @functools.wraps(fn)
        def _wrapper(res: Optional[V]) -> V:
            # TODO how to ensure result state gets reset on each new application?
            return interpreter({get_result: lambda: res})(fn)(res)

        return _wrapper

    def _decorator(fn: Callable[Q, V]) -> Callable[Q, V]:
        return shallow_interpreter(_bind_local_state(unbound_conts))(_init_local_state(fn))

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

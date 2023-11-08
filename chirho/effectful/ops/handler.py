import contextlib
import functools
from typing import Callable, Concatenate, Mapping, Optional, ParamSpec, TypeVar

from chirho.effectful.ops.interpretation import Interpretation, interpreter, shallow_interpreter
from chirho.effectful.ops.operation import Operation, define

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


LocalState = tuple[Optional[T], tuple, dict]


def bind_and_push_prompts(
    unbound_conts: Mapping[Operation[[Optional[S]], S], Callable[[Optional[T], Optional[S]], T]],
) -> Callable[[Callable[Concatenate[Optional[T], P], T]], Callable[Concatenate[Optional[T], P], T]]:

    @define(Operation)
    def get_local_state() -> LocalState[T]:
        raise ValueError("No args stored")

    def _init_local_state(
        fn: Callable[Concatenate[Optional[V], Q], V]
    ) -> Callable[Concatenate[Optional[V], Q], V]:

        @functools.wraps(fn)
        def _wrapper(__res: Optional[V], *a: Q.args, **ks: Q.kwargs) -> V:
            return interpreter({get_local_state: lambda _: (__res, a, ks)})(fn)(__res, *a, **ks)

        return _wrapper

    def _update_local_state(
        fn: Callable[[Optional[T], Optional[T]], T]
    ) -> Callable[[Optional[T], Optional[T]], T]:

        @functools.wraps(fn)
        def _wrapper(r: Optional[T], __res: Optional[T]) -> T:
            updated_state = (__res,) + get_local_state()[1:]
            return interpreter({get_local_state: lambda _: updated_state})(fn)(r, __res)

        return _wrapper

    def _bind_local_state(
        unbound_conts: Mapping[Operation[[Optional[V]], V], Callable[Concatenate[Optional[T], Q], T]],
    ) -> Mapping[Operation[[Optional[V]], V], Callable[[Optional[T], Optional[T]], T]]:
        return {
            p: _update_local_state(functools.partial(
                lambda k, _, __: k(get_local_state()[0], *get_local_state()[1], **get_local_state()[2]),
                unbound_conts[p],
            )) for p in unbound_conts.keys()
        }

    def _decorator(fn: Callable[Concatenate[Optional[V], Q], V]):
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

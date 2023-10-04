import contextlib
import functools
from typing import Callable, Concatenate, Mapping, Optional, ParamSpec, Protocol, TypeVar

from chirho.effectful.ops.interpretation import Interpretation, interpreter
from chirho.effectful.ops.operation import Operation, define
from chirho.effectful.ops.runtime import get_interpretation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")

S_ = TypeVar("S_", contravariant=True)


class Continuation(Protocol[S_, T]):
    def __call__(self, __result: Optional[T], __value: Optional[S_]) -> T:
        ...


@contextlib.contextmanager
def shallow_interpreter(intp: Interpretation):
    # destructive update: calling any op in intp should remove intp from active
    active_intp = get_interpretation()
    prev_intp = {
        op: active_intp[op] if op in active_intp else op.default for op in intp.keys()
    }

    with interpreter({op: interpreter(prev_intp, unset=False)(intp[op]) for op in intp.keys()}):
        yield intp


def capture_cont_args(
    get_args: Operation[[], tuple[tuple, dict]],
    fn: Callable[Concatenate[Optional[T], Q], T]
) -> Callable[Concatenate[Optional[T], Q], T]:

    @functools.wraps(fn)
    def _wrapper(__result: Optional[T], *args: Q.args, **kwargs: Q.kwargs) -> T:
        return interpreter({get_args: lambda _: (args, kwargs)})(fn)(__result, *args, **kwargs)

    return _wrapper


def bind_cont_args(
    get_args: Operation[[], tuple[tuple, dict]],
    unbound_conts: Mapping[Operation[[Optional[S]], S], Callable[Concatenate[Optional[T], P], T]],
) -> Interpretation[S, T]:
    return {
        p: functools.partial(
            lambda k, _, res: k(res, *get_args()[0], **get_args()[1]),
            unbound_conts[p],
        ) for p in unbound_conts.keys()
    }


def bind_and_push_prompts(
    unbound_conts: Interpretation[S, T],
) -> Callable[[Callable[P, T]], Callable[P, T]]:

    def _decorator(fn: Callable[P, T]) -> Callable[P, T]:

        @define(Operation)
        def get_args() -> tuple[tuple, dict]:
            raise ValueError(f"No args stored for {fn}")

        bound_arg_conts = bind_cont_args(get_args, unbound_conts)
        return shallow_interpreter(bound_arg_conts)(capture_cont_args(get_args, fn))

    return _decorator

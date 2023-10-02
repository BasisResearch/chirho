import contextlib
import functools
from typing import Callable, Concatenate, Optional, ParamSpec, Protocol, TypeVar

from chirho.effectful.ops._utils import weak_memoize
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
    prev_intp = define(Interpretation)({
        op: active_intp[op] if op in active_intp else op.default for op in intp.keys()
    })

    with interpreter({op: interpreter(prev_intp, unset=False)(intp[op]) for op in intp.keys()}):
        yield intp


@weak_memoize
def get_cont_args(op: Operation) -> Operation[[], tuple[tuple, dict]]:
    def _null_op():
        raise ValueError(f"No args stored for {op}")

    return define(Operation)(_null_op)


def capture_cont_args(
    op: Operation[P, S], op_intp: Callable[Concatenate[Optional[T], Q], T]
) -> Callable[Concatenate[Optional[T], Q], T]:
    @functools.wraps(op_intp)
    def _wrapper(__result: Optional[T], *args: Q.args, **kwargs: Q.kwargs) -> T:
        return interpreter(
            define(Interpretation)({get_cont_args(op): lambda _: (args, kwargs)})
        )(op_intp)(__result, *args, **kwargs)

    return _wrapper


def bind_cont_args(
    op: Operation[P, S],
    unbound_conts: Interpretation[S, T],
) -> Interpretation[S, T]:
    return define(Interpretation)(
        {
            p: functools.partial(
                lambda k, _, res: k(
                    res, *get_cont_args(op)()[0], **get_cont_args(op)()[1]
                ),
                unbound_conts[p],
            )
            for p in unbound_conts.keys()
        }
    )


def bind_and_push_prompts(
    unbound_conts: Interpretation[S, T],
    op: Operation[P, S],
    op_intp: Callable[Concatenate[Optional[T], P], T],
) -> Callable[Concatenate[Optional[T], P], T]:
    bound_arg_conts = bind_cont_args(op, unbound_conts)
    return shallow_interpreter(bound_arg_conts)(capture_cont_args(op, op_intp))

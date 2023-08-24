import contextlib
import functools
from typing import Callable, Concatenate, Optional, ParamSpec, Protocol, TypeVar

from chirho.effectful.internals.base_continuation import _BaseAffineContinuation
from chirho.effectful.internals.runtime import weak_memoize
from chirho.effectful.ops.interpretation import Interpretation, interpreter, register
from chirho.effectful.ops.operation import Operation, define

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


class Continuation(Protocol[T]):
    def __call__(self, result: Optional[T], value: Optional[T]) -> T:
        ...


class AffineContinuationError(Exception):
    pass


@define(Operation)
@contextlib.contextmanager
def push_prompts(conts: Interpretation[T]):
    from chirho.effectful.internals.runtime import get_interpretation

    resets = define(Interpretation)(
        {
            p: define(Continuation)(
                interpreter(
                    define(Interpretation)(
                        {
                            p: get_interpretation()[p]
                            if p in get_interpretation()
                            else p.default
                        }
                    )
                )(conts[p])
            )
            for p in conts.keys()
        }
    )
    with interpreter(resets):
        yield


@weak_memoize
def get_cont_args(op: Operation[..., T]) -> Operation[[], tuple[tuple, dict]]:
    def _null_op():
        raise ValueError(f"No args stored for {op}")

    return define(Operation)(_null_op)


@define(Operation)
def capture_cont_args(
    op: Operation[P, T], op_intp: Callable[Concatenate[Optional[T], P], T]
) -> Callable[Concatenate[Optional[T], P], T]:
    @functools.wraps(op_intp)
    def _wrapper(result: Optional[T], *args: P.args, **kwargs: P.kwargs) -> T:
        return interpreter(
            define(Interpretation)({get_cont_args(op): lambda _: (args, kwargs)})
        )(op_intp)(result, *args, **kwargs)

    return _wrapper


@define(Operation)
def bind_cont_args(
    op: Operation[P, T],
    unbound_conts: Interpretation[T],
) -> Interpretation[T]:
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


@define(Operation)
def bind_and_push_prompts(
    unbound_conts: Interpretation[T],
    op: Operation[P, T],
    op_intp: Callable[Concatenate[Optional[T], P], T],
) -> Callable[Concatenate[Optional[T], P], T]:
    return push_prompts(bind_cont_args(op, unbound_conts))(
        capture_cont_args(op, op_intp)
    )


# bootstrap
register(define(Continuation), None, _BaseAffineContinuation)

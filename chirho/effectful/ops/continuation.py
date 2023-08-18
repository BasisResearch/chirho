import contextlib
import functools
from typing import Callable, Concatenate, Generic, Optional, ParamSpec, TypeVar

from chirho.effectful.internals.runtime import get_interpretation, weak_memoize
from chirho.effectful.ops.interpretation import Interpretation, interpreter
from chirho.effectful.ops.operation import Operation, define

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


Continuation = Callable[[Optional[T], Optional[T]], T]


class AffineContinuationError(Exception):
    pass


class _AffineContinuation(Generic[T]):
    cont: Continuation[T]
    used: bool

    def __init__(self, cont: Continuation[T]):
        self.cont = cont
        self.used = False

    def __call__(self, _res: Optional[T], value: Optional[T]) -> T:
        try:
            if self.used:
                raise AffineContinuationError(f"can use {self.cont} at most once")
            return self.cont(_res, value)
        finally:
            self.used = True


@define(Operation)
@contextlib.contextmanager
def push_prompts(conts: Interpretation[T]):
    resets = define(Interpretation)(
        {
            p: _AffineContinuation(
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

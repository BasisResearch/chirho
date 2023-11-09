import collections.abc
import contextlib
import functools
import typing
from typing import Callable, Concatenate, Optional, ParamSpec, TypeVar

from chirho.effectful.ops.operation import Operation, define

from ..internals.base_interpretation import _BaseInterpretation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")

Interpretation = collections.abc.Mapping[Operation[..., T], Callable[..., V]]


@contextlib.contextmanager
def interpreter(intp: Interpretation, *, unset: bool = True):
    from .runtime import get_interpretation, swap_interpretation

    old_intp = get_interpretation()
    try:
        new_intp = {
            op: intp[op] if op in intp else old_intp[op]
            for op in set(intp.keys()) | set(old_intp.keys())
        }
        old_intp = swap_interpretation(new_intp)
        yield intp
    finally:
        if unset:
            _ = swap_interpretation(old_intp)
        else:
            if len(list(old_intp.keys())) == 0 and len(list(intp.keys())) > 0:
                raise RuntimeError(f"Dangling interpretation on stack: {intp}")


@contextlib.contextmanager
def shallow_interpreter(intp: Interpretation):
    from .runtime import get_interpretation

    # destructive update: calling any op in intp should remove intp from active
    active_intp = get_interpretation()
    prev_intp = {
        op: active_intp[op] if op in active_intp else op.default for op in intp.keys()
    }

    with interpreter({op: interpreter(prev_intp, unset=False)(intp[op]) for op in intp.keys()}):
        yield intp


@define(Operation)
def get_result() -> Optional[T]:
    return None


def bind_result(fn: Callable[Concatenate[Optional[T], P], T]) -> Callable[P, T]:

    @functools.wraps(fn)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return fn(get_result(), *args, **kwargs)

    return _wrapper


@define(Operation)
def register_(
    __op: Operation[P, T],
    intp: Optional[Interpretation[T, V]],
    interpret_op: Callable[Q, V],
) -> Callable[Q, V]:
    if intp is None:
        setattr(
            __op,
            "default",
            interpret_op,  # functools.wraps(__op.default)(interpret_op),
        )
        return interpret_op
    elif isinstance(intp, collections.abc.MutableMapping):
        intp.__setitem__(__op, interpret_op)
        return interpret_op
    raise NotImplementedError(f"Cannot register {__op} in {intp}")


@typing.overload
def register(
    __op: Operation[P, T],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    ...


@typing.overload
def register(
    __op: Operation[P, T],
    intp: Optional[Interpretation[T, V]],
) -> Callable[[Callable[Q, V]], Callable[Q, V]]:
    ...


@typing.overload
def register(
    __op: Operation[P, T],
    intp: Optional[Interpretation[T, V]],
    interpret_op: Callable[Q, V],
) -> Callable[Q, V]:
    ...


def register(__op, intp=None, interpret_op=None):
    if interpret_op is None:
        return lambda interpret_op: register(__op, intp, interpret_op)
    else:
        return register_(__op, intp, interpret_op)


# bootstrap
register(define(Interpretation), None, _BaseInterpretation)

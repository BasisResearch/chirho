import collections.abc
import contextlib
import functools
import typing
from typing import Callable, Concatenate, Mapping, Optional, ParamSpec, Tuple, TypeVar

from chirho.meta.ops.operation import Operation, define

from ..internals.base_interpretation import _BaseInterpretation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")

Interpretation = collections.abc.Mapping[Operation[..., T], Callable[..., V]]


@typing.overload
def register(op: Operation[P, T]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    ...


@typing.overload
def register(
    op: Operation[P, T],
    intp: Optional[Interpretation[T, V]],
) -> Callable[[Callable[Q, V]], Callable[Q, V]]:
    ...


@typing.overload
def register(
    op: Operation[P, T],
    intp: Optional[Interpretation[T, V]],
    interpret_op: Callable[Q, V],
) -> Callable[Q, V]:
    ...


def register(op, intp=None, interpret_op=None):
    if interpret_op is None:
        return lambda interpret_op: register(op, intp, interpret_op)

    if intp is None:
        setattr(op, "default", interpret_op)
        return interpret_op
    elif isinstance(intp, collections.abc.MutableMapping):
        intp.__setitem__(op, interpret_op)
        return interpret_op
    raise NotImplementedError(f"Cannot register {op} in {intp}")


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

    with interpreter(
        {op: interpreter(prev_intp, unset=False)(intp[op]) for op in intp.keys()}
    ):
        yield intp


@define(Operation)
def _get_result() -> Optional[T]:
    return None


def _set_result(
    fn: Callable[Concatenate[Optional[T], P], T]
) -> Callable[Concatenate[Optional[T], P], T]:
    @functools.wraps(fn)
    def _wrapper(res: Optional[T], *args: P.args, **kwargs: P.kwargs) -> T:
        return shallow_interpreter({_get_result: lambda: res})(fn)(res, *args, **kwargs)

    return _wrapper


def bind_result(fn: Callable[Concatenate[Optional[T], P], T]) -> Callable[P, T]:
    @functools.wraps(fn)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return interpreter({_get_result: _get_result.default})(fn)(
            _get_result(), *args, **kwargs
        )

    return _wrapper


Prompt = Operation[[Optional[T]], T]


def bind_prompts(
    unbound_conts: Mapping[Prompt[S], Callable[P, T]],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    LocalState = Tuple[Tuple, Mapping]

    @define(Operation)
    def _get_local_state() -> LocalState:
        raise ValueError("No args stored")

    def _set_local_state(fn: Callable[Q, V]) -> Callable[Q, V]:
        @functools.wraps(fn)
        def _wrapper(*a: Q.args, **ks: Q.kwargs) -> V:
            return interpreter({_get_local_state: lambda: (a, ks)})(fn)(*a, **ks)

        return _wrapper

    def _bind_local_state(fn: Callable[Q, V]) -> Callable[Q, V]:
        bound_conts = {
            p: _set_result(
                functools.partial(
                    lambda k, _: k(*_get_local_state()[0], **_get_local_state()[1]),
                    unbound_conts[p],
                )
            )
            for p in unbound_conts.keys()
        }
        return shallow_interpreter(bound_conts)(_set_local_state(fn))

    return _bind_local_state


# bootstrap
register(define(Interpretation), None, _BaseInterpretation)

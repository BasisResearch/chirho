import contextlib
import functools
import typing
from typing import Callable, Concatenate, Iterable, Optional, ParamSpec, Protocol, TypeVar

from chirho.effectful.ops.operation import Operation, define

from ..internals.base_interpretation import _BaseInterpretation

P = ParamSpec("P")
Q = ParamSpec("Q")
T = TypeVar("T")
V = TypeVar("V")


@typing.runtime_checkable
class Interpretation(Protocol[T, V]):
    def __setitem__(
        self, __op: Operation[P, T], __interpret: Callable[Concatenate[Optional[V], Q], V]
    ) -> None:
        ...

    def __getitem__(
        self, __op: Operation[P, T]
    ) -> Callable[Concatenate[Optional[V], Q], V]:
        ...

    def __contains__(self, __op: Operation[..., T]) -> bool:
        ...

    def keys(self) -> Iterable[Operation[..., T]]:
        ...


@typing.overload
def register(
    __op: Operation[P, T],
) -> Callable[[Callable[Concatenate[Optional[T], P], T]], Callable[Concatenate[Optional[T], P], T]]:
    ...


@typing.overload
def register(
    __op: Operation[P, T],
    intp: Optional[Interpretation[T, V]],
) -> Callable[[Callable[Concatenate[Optional[V], Q], V]], Callable[Concatenate[Optional[V], Q], V]]:
    ...


@typing.overload
def register(
    __op: Operation[P, T],
    intp: Optional[Interpretation[T, V]],
    interpret_op: Callable[Concatenate[Optional[V], Q], V],
) -> Callable[Concatenate[Optional[V], Q], V]:
    ...


@define(Operation)
def register(__op, intp=None, interpret_op=None):
    if interpret_op is None:
        return lambda interpret_op: register(__op, intp, interpret_op)

    if intp is None:
        setattr(
            __op,
            "default",
            functools.wraps(__op.default)(
                lambda result, *args, **kwargs: interpret_op(*args, **kwargs)
                if result is None
                else result
            ),
        )
        return interpret_op
    elif isinstance(intp, Interpretation):
        intp.__setitem__(__op, interpret_op)
        return interpret_op
    raise NotImplementedError(f"Cannot register {__op} in {intp}")


@define(Operation)
@contextlib.contextmanager
def interpreter(intp: Interpretation):
    from ..internals.runtime import get_interpretation, swap_interpretation

    old_intp = get_interpretation()
    try:
        new_intp = define(Interpretation)(
            {
                op: intp[op] if op in intp else old_intp[op]
                for op in set(intp.keys()) | set(old_intp.keys())
            }
        )
        old_intp = swap_interpretation(new_intp)
        yield intp
    finally:
        swap_interpretation(old_intp)


# bootstrap
register(define(Interpretation), None, _BaseInterpretation)

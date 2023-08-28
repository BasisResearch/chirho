import contextlib
import functools
import typing
from typing import (
    Callable,
    ClassVar,
    Concatenate,
    Dict,
    Generic,
    Iterable,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
)

from chirho.effectful.ops.operation import Operation, define

from ..internals.base_interpretation import _BaseInterpretation

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Interpretation(Protocol[T]):
    def __setitem__(self, op: Operation[P, T], interpret: Callable[P, T]) -> None:
        ...

    def __getitem__(self, op: Operation[P, T]) -> Callable[P, T]:
        ...

    def __contains__(self, op: Operation[..., T]) -> bool:
        ...

    def keys(self) -> Iterable[Operation[..., T]]:
        ...


class StatefulInterpretation(Generic[S, T]):
    state: S

    _op_intps: ClassVar[Dict[Operation, Callable]] = {}

    def __init_subclass__(cls) -> None:
        cls._op_intps = {}
        return super().__init_subclass__()

    def __init__(self, state: S):
        self.state = state

    @classmethod
    def __setitem__(
        cls,
        op: Operation[P, T],
        interpret_op: Callable[Concatenate[S, P], T],
    ) -> None:
        cls._op_intps[op] = interpret_op

    @classmethod
    def __contains__(cls, op: Operation[..., T]) -> bool:
        return op in cls._op_intps

    def __getitem__(self, op: Operation[P, T]) -> Callable[P, T]:
        op_intp: Callable[Concatenate[S, P], T] = self._op_intps[op]
        return functools.partial(op_intp, self.state)

    @classmethod
    def keys(cls) -> Iterable[Operation[..., T]]:
        return cls._op_intps.keys()


@typing.overload
def register(
    op: Operation[P, T],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    ...


@typing.overload
def register(
    op: Operation[P, T],
    intp: Optional[Interpretation[T]],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    ...


@typing.overload
def register(
    op: Operation[P, T],
    intp: Optional[Interpretation[T]],
    interpret_op: Callable[P, T],
) -> Callable[P, T]:
    ...


@define(Operation)
def register(op, intp=None, interpret_op=None):
    if interpret_op is None:
        return lambda interpret_op: register(op, intp, interpret_op)

    if intp is None:
        setattr(op, "default", interpret_op)
        return interpret_op
    elif isinstance(intp, Interpretation):
        intp.__setitem__(op, interpret_op)
        return interpret_op
    raise NotImplementedError(f"Cannot register {op} in {intp}")


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

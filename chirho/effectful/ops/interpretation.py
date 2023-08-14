import contextlib
import functools
import typing
from typing import Callable, ClassVar, Generic, Iterable, Optional, Protocol, TypeVar

from chirho.effectful.ops.operation import Operation, define

from ..internals.base_interpretation import _BaseInterpretation

S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Interpretation(Protocol[T]):
    def __setitem__(self, op: Operation[T], interpret: Callable[..., T]) -> None:
        ...

    def __getitem__(self, op: Operation[T]) -> Callable[..., T]:
        ...

    def __contains__(self, op: Operation[T]) -> bool:
        ...

    def keys(self) -> Iterable[Operation[T]]:
        ...


class StatefulInterpretation(Generic[S, T]):
    state: S

    _op_interpretations: ClassVar[dict[Operation, Callable]] = {}

    def __init_subclass__(cls) -> None:
        cls._op_interpretations = {}
        return super().__init_subclass__()

    def __init__(self, state: S):
        self.state = state

    @classmethod
    def __setitem__(cls, op: Operation[T], interpret_op: Callable[..., T]) -> None:
        cls._op_interpretations[op] = interpret_op

    @classmethod
    def __contains__(cls, op: Operation[T]) -> bool:
        return op in cls._op_interpretations

    def __getitem__(self, op: Operation[T]) -> Callable[..., T]:
        return functools.partial(self._op_interpretations[op], self.state)

    @classmethod
    def keys(cls) -> Iterable[Operation[T]]:
        return cls._op_interpretations.keys()


@define(Operation)
def register(
    op: Operation[T],
    intp: Optional[Interpretation[T]] = None,
    interpret_op: Optional[Callable[..., T]] = None,
):
    if interpret_op is None:
        return lambda interpret_op: register(op, intp, interpret_op)

    if intp is None:
        op.default = functools.wraps(op.default)(
            lambda result, *args, **kwargs: interpret_op(*args, **kwargs)
            if result is None
            else result
        )
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
register(define(Interpretation))(_BaseInterpretation)

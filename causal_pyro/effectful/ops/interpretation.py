from typing import Callable, ClassVar, Generic, Hashable, Iterable, List, Mapping, Optional, Protocol, Type, TypedDict, TypeVar, runtime_checkable

import contextlib
import functools

S = TypeVar("S")
T = TypeVar("T")


@runtime_checkable
class Operation(Protocol[T]):
    def __call__(self, *args: T, **kwargs) -> T: ...


@runtime_checkable
class Interpretation(Protocol[T]):
    def __setitem__(self, op: Operation[T], interpret: Callable[..., T]) -> None: ...
    def __getitem__(self, op: Operation[T]) -> Callable[..., T]: ...
    def __contains__(self, op: Operation[T]) -> bool: ...
    def keys(self) -> Iterable[Operation[T]]: ...


@functools.cache
def define(m: Type[T]) -> Operation[T]:
    if m is Operation:
        from ._runtime import _BaseOperation, swap_interpretation, _op_call
        swap_interpretation(define(Interpretation)())
        setattr(_BaseOperation, "__call__", _op_call)
        return _BaseOperation(_BaseOperation)
    elif m is Interpretation:
        from ._runtime import _BaseOperation, _BaseInterpretation
        return _BaseOperation(_BaseInterpretation)
    else:
        return define(Operation)(m)


# triggers bootstrapping of Operation, Interpretation
define = define(Operation)(define)


@define(Operation)
def register(
    op: Operation[T],
    intp: Optional[Interpretation[T]] = None,
    interpret_op: Optional[Callable[..., T]] = None
):
    if interpret_op is None:
        return lambda interpret_op: register(op, intp, interpret_op)

    if intp is None:
        if isinstance(op, Operation):
            setattr(op, "body", interpret_op)  # TODO resolve confusion of body, default
            return interpret_op
    elif isinstance(intp, Interpretation):
        intp.__setitem__(op, interpret_op)
        return interpret_op
    raise NotImplementedError(f"Cannot register {op} in {intp}")


@define(Operation)
def union(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:
        return intp
    elif len(intps) == 1:
        intp2, = intps
        return define(Interpretation)({
            op: intp2[op] if op in intp2 else intp[op]
            for op in set(intp.keys()) | set(intp2.keys())
        })
    else:
        return union(intp, *intps)


@define(Operation)
@contextlib.contextmanager
def interpreter(intp: Interpretation[T]):
    from ._runtime import get_interpretation, swap_interpretation

    old_intp = swap_interpretation(union(get_interpretation(), intp))
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)

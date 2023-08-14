import typing
from typing import Optional, Protocol, Type, TypeVar

from ..internals import runtime

S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Operation(Protocol[T]):
    def __call__(self, *args, **kwargs) -> T:
        ...

    def default(self, result: Optional[T], *args, **kwargs) -> T:
        ...


@typing.overload
def define(m: Type[Operation[T]]) -> Operation[Operation[T]]:
    ...


@typing.overload
def define(m: Type[T]) -> Operation[T]:
    ...


def define(m):
    """
    Scott encoding of a type as its constructor.
    """
    if typing.get_origin(m) not in (m, None):
        return define(typing.get_origin(m))

    if m is Operation:
        from ..internals.base_operation import _BaseOperation
        return _BaseOperation[Operation[m]](_BaseOperation[m])

    defop: Operation[Operation[m]] = define(Operation)
    return defop(typing.get_origin(m) if typing.get_origin(m) is not None else m)


# triggers bootstrapping
define = define(Operation)(runtime.weak_memoize(define))
runtime.get_interpretation = define(Operation)(runtime.get_interpretation)
runtime.swap_interpretation = define(Operation)(runtime.swap_interpretation)

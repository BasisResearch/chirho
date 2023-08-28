import typing
from typing import ParamSpec, Protocol, Type, TypeVar

from ..internals import runtime

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T", covariant=True)


@typing.runtime_checkable
class Operation(Protocol[P, T]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        ...

    def default(self, *args: P.args, **kwargs: P.kwargs) -> T:
        ...


def define(m: Type[T]) -> Operation[P, T]:
    """
    Scott encoding of a type as its constructor.
    """
    if not typing.TYPE_CHECKING:
        if typing.get_origin(m) not in (m, None):
            return define(typing.get_origin(m))

    if m is Operation:
        from ..internals.base_operation import _BaseOperation

        return _BaseOperation(_BaseOperation)

    defop: Operation[..., Operation[P, T]] = define(Operation[P, T])
    return defop(m)


# triggers bootstrapping
define = define(Operation)(runtime.weak_memoize(define))
runtime.get_interpretation = define(Operation)(runtime.get_interpretation)
runtime.swap_interpretation = define(Operation)(runtime.swap_interpretation)

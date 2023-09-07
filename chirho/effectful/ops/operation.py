import typing
from typing import Callable, Concatenate, Mapping, Optional, ParamSpec, Protocol, Type, TypeVar

from ..internals import runtime

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Operation(Protocol[P, T]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        ...

    def default(self, __result: Optional[T], *args: P.args, **kwargs: P.kwargs) -> T:
        ...


def apply(
    interpretation: Mapping[Operation[P, T], Callable[Concatenate[Optional[S], Q], S]],
    op: Operation[P, T],
    *args: Q.args,
    **kwargs: Q.kwargs
) -> S:

    try:
        interpret = interpretation[op]
    except KeyError:
        interpret = op.default
    return interpret(None, *args, **kwargs)


@runtime.weak_memoize
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

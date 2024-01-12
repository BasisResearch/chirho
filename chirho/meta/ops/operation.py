import typing
from typing import Callable, Mapping, ParamSpec, Protocol, Type, TypeGuard, TypeVar

from ._utils import weak_memoize

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@typing.runtime_checkable
class Operation(Protocol[P, T_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        ...

    def default(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        ...


@weak_memoize
def define(m: Type[T] | Callable[Q, T]) -> Operation[..., T]:
    """
    Scott encoding of a type as its constructor.
    """
    if not typing.TYPE_CHECKING:
        if typing.get_origin(m) not in (m, None):
            return define(typing.get_origin(m))

    def _is_op_type(m: Type[S] | Callable[P, S]) -> TypeGuard[Type[Operation[..., S]]]:
        return typing.get_origin(m) is Operation or m is Operation

    if _is_op_type(m):
        from ..internals.base_operation import _BaseOperation

        @_BaseOperation
        def defop(fn: Callable[..., S]) -> _BaseOperation[..., S]:
            return _BaseOperation(fn)

        return defop
    else:
        return define(Operation[..., T])(m)


def apply(
    interpretation: Mapping[Operation[P, T], Callable[P, S]],
    op: Operation[P, T],
    *args: P.args,
    **kwargs: P.kwargs
) -> S:
    try:
        interpret = interpretation[op]
    except KeyError:
        interpret = op.default
    return interpret(*args, **kwargs)

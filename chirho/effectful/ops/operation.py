import typing
from typing import Callable, Mapping, ParamSpec, Protocol, Type, TypeVar

from ._utils import weak_memoize

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Operation(Protocol[P, T]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        ...

    def default(self, *args: P.args, **kwargs: P.kwargs) -> T:
        ...


def apply(
    interpretation: Mapping[Operation[P, T], Callable[Q, S]],
    op: Operation[P, T],
    *args: Q.args,
    **kwargs: Q.kwargs
) -> S:

    try:
        interpret = interpretation[op]
    except KeyError:
        interpret = op.default
    return interpret(*args, **kwargs)


@typing.overload
def define(m: Type[T] | Callable[Q, T]) -> Operation[P, T]:
    ...


@typing.overload
def define(m: Type[Operation[Q, S]]) -> Operation[[Callable[Q, S]], Operation[Q, S]]:
    ...


def define(m):
    """
    Scott encoding of a type as its constructor.
    """
    if not typing.TYPE_CHECKING:
        if typing.get_origin(m) not in (m, None):
            return define(typing.get_origin(m))

    if m is Operation:
        from ..internals.base_operation import _BaseOperation

        return _BaseOperation(_BaseOperation)

    return define(Operation)(m)


if not typing.TYPE_CHECKING:
    define = weak_memoize(define)
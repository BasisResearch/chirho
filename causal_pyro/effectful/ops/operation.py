from typing import Generic, Callable, Optional, Protocol, Type, TypeVar

import typing

from ..internals import runtime


S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Operation(Protocol[T]):
    def __call__(self, *args, **kwargs) -> T: ...
    def default(self, result: Optional[T], *args, **kwargs) -> T: ...


class _BaseOperation(Generic[T]):

    def __init__(self, body: Callable[..., T]):
        self._body = body

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({getattr(self._body, '__name__', self._body)})"

    def default(self, result: Optional[T], *args, **kwargs) -> T:
        return result if result is not None else self._body(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> T:
        intp = runtime.get_runtime()["interpretation"] \
            if self is runtime.get_interpretation \
            else runtime.get_interpretation()
        try:
            interpret = intp[self]
        except KeyError:
            interpret = self.default
        return interpret(None, *args, **kwargs)


@typing.overload
def define(m: Type[Operation[T]]) -> Operation[Operation[T]]: ...


@typing.overload
def define(m: Type[T]) -> Operation[T]: ...


def define(m):
    """
    Scott encoding of a type as its constructor.
    """
    if typing.get_origin(m) not in (m, None):
        return define(typing.get_origin(m))

    if m is Operation:
        return _BaseOperation[Operation[m]](_BaseOperation[m])

    defop: Operation[Operation[m]] = define(Operation)
    return defop(typing.get_origin(m) if typing.get_origin(m) is not None else m)


# triggers bootstrapping
define = define(Operation)(runtime.weak_memoize(define))
runtime.get_interpretation = define(Operation)(runtime.get_interpretation)
runtime.swap_interpretation = define(Operation)(runtime.swap_interpretation)

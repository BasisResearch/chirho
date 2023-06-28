from typing import Generic, Callable, Optional, Protocol, Type, TypeVar

import functools
import typing

from ..internals import runtime


S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Operation(Protocol[T]):
    def __call__(self, *args: T, **kwargs) -> T: ...
    def default(self, result: Optional[T], *args: T, **kwargs) -> T: ...


class _BaseOperation(Generic[T]):

    def __init__(self, body: Callable[..., T]):
        self.body = body

    @property
    def default(self) -> Callable[..., T]:
        return functools.wraps(self.body)(
            lambda res, *args, **kwargs: res if res is not None else self.body(*args, **kwargs)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({getattr(self.body, '__name__', self.body)}"

    def __call__(self, *args: T, **kwargs: T) -> T:
        intp = runtime.get_runtime()["interpretation"] \
            if self is runtime.get_interpretation \
            else runtime.get_interpretation()
        try:
            interpret = intp[self]
        except KeyError:
            interpret = self.default  # TODO abstract or codify default?
        return interpret(None, *args, **kwargs)


@functools.cache
def define(m: Type[T]) -> Operation[T]:
    """
    Scott encoding of a type as its constructor.
    """

    if typing.get_origin(m) is Operation:

        @_BaseOperation
        def _defop(op: Callable[..., T]) -> Operation[T]:
            return functools.wraps(op)(_BaseOperation(op))

        return _defop  # _BaseOperation(_BaseOperation)

    return define(Operation[m])(m)


# triggers bootstrapping
define = define(Operation)(define)
runtime.get_interpretation = define(Operation)(runtime.get_interpretation)
runtime.swap_interpretation = define(Operation)(runtime.swap_interpretation)

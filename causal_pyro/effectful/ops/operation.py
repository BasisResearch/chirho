from typing import Generic, Callable, Protocol, Type, TypeVar

import functools
import typing

from ..internals import runtime


S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Operation(Protocol[T]):
    def __call__(self, *args: T, **kwargs) -> T: ...


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
    if not typing.TYPE_CHECKING:
        if typing.get_args(m):
            return define(typing.get_origin(m))

    if m is Operation:
        # return _BaseOperation(_BaseOperation)
        return _BaseOperation(lambda op: functools.wraps(op)(_BaseOperation(op)))

    return define(Operation)(m)


# triggers bootstrapping
define = define(Operation)(define)
runtime.get_interpretation = define(Operation)(runtime.get_interpretation)
runtime.swap_interpretation = define(Operation)(runtime.swap_interpretation)

from typing import Callable, Generic, Optional, TypeVar

from ..internals import runtime

S = TypeVar("S")
T = TypeVar("T")


class _BaseOperation(Generic[T]):
    def __init__(self, body: Callable[..., T]):
        self._body = body

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({getattr(self._body, '__name__', self._body)})"
        )

    def default(self, result: Optional[T], *args, **kwargs) -> T:
        return result if result is not None else self._body(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> T:
        intp = (
            runtime.get_runtime().interpretation
            if self is runtime.get_interpretation
            else runtime.get_interpretation()
        )
        try:
            interpret = intp[self]
        except KeyError:
            interpret = self.default
        return interpret(None, *args, **kwargs)

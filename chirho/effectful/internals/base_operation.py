from typing import Callable, Generic, ParamSpec, TypeVar

from ..internals import runtime

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


class _BaseOperation(Generic[P, T]):
    def __init__(self, body: Callable[P, T]):
        self._body = body

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({getattr(self._body, '__name__', self._body)})"
        )

    def default(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self._body(*args, **kwargs)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        intp = (
            runtime.get_runtime().interpretation
            if self is runtime.get_interpretation
            else runtime.get_interpretation()
        )
        try:
            interpret = intp[self]
        except KeyError:
            interpret = self.default
        return interpret(*args, **kwargs)

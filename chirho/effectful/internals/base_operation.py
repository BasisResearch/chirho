from typing import Callable, Generic, Optional, ParamSpec, TypeVar

from ..ops import runtime
from ..ops import operation

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


class _BaseOperation(Generic[P, T]):
    def __init__(self, __body: Callable[P, T]):
        self._body = __body

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({getattr(self._body, '__name__', self._body)})"
        )

    def default(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self._body(*args, **kwargs)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if self is runtime.get_runtime:
            intp = getattr(runtime.get_runtime, "default")(*args, **kwargs).interpretation
            return operation.apply.default(intp, self, *args, **kwargs)
        elif self is operation.apply:
            intp = runtime.get_interpretation()
            return operation.apply.default(intp, self, *args, **kwargs)
        else:
            intp = runtime.get_interpretation()
            return operation.apply(intp, self, *args, **kwargs)


runtime.get_runtime = _BaseOperation(runtime.get_runtime)
operation.apply = _BaseOperation(operation.apply)
operation.define = operation.define(operation.Operation)(operation.define)

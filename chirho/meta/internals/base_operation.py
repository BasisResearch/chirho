from typing import Callable, Generic, ParamSpec, TypeVar

from ..ops import operation, runtime

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class _BaseOperation(Generic[P, T_co]):
    default: Callable[P, T_co]

    def __init__(self, __default: Callable[P, T_co]):
        self.default = __default

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({getattr(self.default, '__name__', self.default)})"

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        if self is runtime.get_runtime:
            intp = getattr(runtime.get_runtime, "default")(
                *args, **kwargs
            ).interpretation
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

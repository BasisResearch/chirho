from typing import Callable, Dict, Generic, ParamSpec, TypeVar

from chirho.effectful.ops.operation import Operation

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


class _BaseInterpretation(Generic[T], Dict[Operation[..., T], Callable[..., T]]):
    pass

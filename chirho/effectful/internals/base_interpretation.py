from typing import Callable, Dict, Generic, TypeVar

from chirho.effectful.ops.operation import Operation

S = TypeVar("S")
T = TypeVar("T")


class _BaseInterpretation(Generic[T], Dict[Operation[T], Callable[..., T]]):
    pass

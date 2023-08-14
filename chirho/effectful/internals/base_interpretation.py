from typing import Callable, Generic, TypeVar

from chirho.effectful.ops.operation import Operation

S = TypeVar("S")
T = TypeVar("T")


class _BaseInterpretation(Generic[T], dict[Operation[T], Callable[..., T]]):
    pass

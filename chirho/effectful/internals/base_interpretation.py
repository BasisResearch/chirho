from typing import Callable, ClassVar, Concatenate, Dict, Generic, Iterable, Optional, ParamSpec, TypeVar

import functools

from chirho.effectful.ops.operation import Operation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


class _BaseInterpretation(Generic[S, T], Dict[Operation[..., S], Callable[..., T]]):
    pass


class _StatefulInterpretation(Generic[S, T, V]):
    state: S

    _op_intps: ClassVar[Dict[Operation, Callable]] = {}

    def __init_subclass__(cls) -> None:
        cls._op_intps = {}
        return super().__init_subclass__()

    def __init__(self, state: S):
        self.state = state

    @classmethod
    def __setitem__(
        cls,
        op: Operation[P, T],
        interpret_op: Callable[Concatenate[S, Optional[V], Q], V],
    ) -> None:
        cls._op_intps[op] = interpret_op

    @classmethod
    def __contains__(cls, op: Operation[..., T]) -> bool:
        return op in cls._op_intps

    def __getitem__(
        self, op: Operation[P, T]
    ) -> Callable[Concatenate[Optional[V], Q], V]:
        return functools.partial(self._op_intps[op], self.state)

    @classmethod
    def keys(cls) -> Iterable[Operation[..., T]]:
        return cls._op_intps.keys()

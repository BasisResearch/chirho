from typing import Callable, ClassVar, Concatenate, Dict, Generic, Iterable, ParamSpec, TypeVar

import functools

from chirho.meta.ops.operation import Operation

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
        __op: Operation[P, T],
        __interpret_op: Callable[Concatenate[S, Q], V],
    ) -> None:
        cls._op_intps[__op] = __interpret_op

    @classmethod
    def __contains__(cls, __op: Operation[..., T]) -> bool:
        return __op in cls._op_intps

    def __getitem__(
        self, __op: Operation[P, T]
    ) -> Callable[Q, V]:
        return functools.partial(self._op_intps[__op], self.state)

    @classmethod
    def keys(cls) -> Iterable[Operation[..., T]]:
        return cls._op_intps.keys()

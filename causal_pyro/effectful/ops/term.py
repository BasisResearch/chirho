from typing import Generic, Iterable, Mapping, Protocol, TypeVar

import typing

from causal_pyro.effectful.ops.environment import Variable
from causal_pyro.effectful.ops.interpretation import Interpretation, register
from causal_pyro.effectful.ops.operation import Operation, define


S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Term(Protocol[T]):
    __head__: Operation[T]
    __args__: Iterable["Term[T]" | Variable[T] | T]
    __kwargs__: Mapping[str, "Term[T]" | Variable[T] | T]


@register(define(Term))
class _BaseTerm(Generic[T], Term[T]):
    __head__: Operation[T]
    __args__: tuple[Term[T] | Variable[T] | T, ...]
    __kwargs__: dict[str, Term[T] | Variable[T] | T]

    def __init__(
        self,
        head: Operation[T],
        args: Iterable[Term[T] | Variable[T] | T],
        kwargs: Mapping[str, Term[T] | Variable[T] | T]
    ):
        self.__head__ = head
        self.__args__ = tuple(args)
        self.__kwargs__ = dict(kwargs)

    def __repr__(self) -> str:
        return f"{self.__head__}(" + \
            f"{', '.join(map(repr, self.__args__))}," + \
            f"{', '.join(f'{k}={v}' for k, v in self.__kwargs__.items())})"


@define(Operation)
def head_of(term: Term[T]) -> Operation[T]:
    return term.__head__


@define(Operation)
def args_of(term: Term[T]) -> Iterable[Term[T] | Variable[T] | T]:
    return term.__args__


@define(Operation)
def kwargs_of(term: Term[T]) -> Mapping[str, Term[T] | Variable[T] | T]:
    return term.__kwargs__


@define(Operation)
def LazyInterpretation(*ops: Operation[T]) -> Interpretation[T | Term[T] | Variable[T]]:
    return define(Interpretation)({
        op: lambda _, *args, **kwargs: define(Term)(op, args, kwargs)
        for op in ops
    })

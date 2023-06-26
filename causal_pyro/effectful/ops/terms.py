from typing import Generic, Iterable, Protocol, TypeVar, runtime_checkable

from .operations import Interpretation, Operation, define, register
from .environments import Variable


S = TypeVar("S")
T = TypeVar("T")


@runtime_checkable
class Term(Protocol[T]):
    __head__: Operation[T]
    __args__: tuple["Term[T]" | Variable[T] | T, ...]


@register(define(Term))
class _BaseTerm(Generic[T], Term[T]):
    __head__: Operation[T]
    __args__: tuple["Term[T]" | Variable[T] | T, ...]

    def __init__(self, head: Operation[T], args: Iterable[Term[T] | Variable[T] | T]):
        self.__head__ = head
        self.__args__ = tuple(args)

    def __repr__(self) -> str:
        return f"{self.__head__}({', '.join(map(repr, self.__args__))})"


@define(Operation)
def head_of(term: Term[T]) -> Operation[T]:
    return term.__head__


@define(Operation)
def args_of(term: Term[T]) -> Iterable[Term[T] | Variable[T] | T]:
    return term.__args__


@define(Operation)
def LazyInterpretation(*ops: Operation[T]) -> Interpretation[T | Term[T] | Variable[T]]:
    return define(Interpretation)({
        op: lambda _, *args: define(Term)(op, args)
        for op in ops
    })

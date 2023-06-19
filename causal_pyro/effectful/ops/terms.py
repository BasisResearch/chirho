from typing import Any, Callable, Container, ContextManager, Generic, Hashable, Iterable, NamedTuple, Optional, Protocol, Set, Type, TypeVar, Union, runtime_checkable

import functools

from .bootstrap import Interpretation, Operation, define, register
from .interpretations import product, reflect, reflections


S, T = TypeVar("S"), TypeVar("T")


class Variable(Protocol[T]):
    __variable_name__: Hashable
    __variable_type__: Optional[Type[T]]


@register(define(Variable))
class BaseVariable(Generic[T], Variable[T]):
    __variable_name__: Hashable
    __variable_type__: Optional[Type[T]]

    def __init__(self, name: Hashable, type: Optional[Type[T]] = None):
        self.__variable_name__ = name
        self.__variable_type__ = type

    def __repr__(self) -> str:
        return f"{self.__variable_name__}: {getattr(self, '__variable_type___', None)}"


class Term(Protocol[T]):
    __head__: Operation[T]
    __args__: tuple["Term[T]" | Variable[T] | T, ...]


@register(define(Term))
class BaseTerm(Generic[T], Term[T]):
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

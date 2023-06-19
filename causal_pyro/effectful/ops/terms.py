from typing import Any, Callable, Container, ContextManager, Generic, Hashable, Iterable, NamedTuple, Optional, Protocol, Set, Type, TypeVar, Union, runtime_checkable

import functools

from .bootstrap import Interpretation, Operation, define, register
from .interpretations import product, reflect, reflections


S, T = TypeVar("S"), TypeVar("T")


class Term(Protocol[T]):
    __head__: Operation[T]
    __args__: tuple["Term[T]" | T, ...]


@register(define(Term))
class BaseTerm(Generic[T], Term[T]):
    __head__: Operation[T]
    __args__: tuple["Term[T]" | T, ...]

    def __init__(self, head: Operation[T], args: Iterable["Term[T]" | T]):
        self.__head__ = head
        self.__args__ = tuple(args)

    def __repr__(self) -> str:
        return f"{self.__head__.__name__}({', '.join(map(repr, self.__args__))})"


@define(Operation)
def head_of(term: Term[T]) -> Operation[T]:
    return term.__head__


@define(Operation)
def args_of(term: Term[T]) -> Iterable[Term[T] | T]:
    return term.__args__


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


LazyVal = T | Term[T] | Variable[T]

@define(Operation)
def LazyInterpretation(*ops: Operation[T]) -> Interpretation[LazyVal[T]]:
    return product(reflections(define(Term)), define(Interpretation)({
        op: functools.partial(define(Term), op) for op in ops
    }))

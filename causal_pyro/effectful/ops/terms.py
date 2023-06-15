from typing import Any, Callable, Container, ContextManager, Generic, Hashable, Iterable, NamedTuple, Optional, Protocol, Set, Type, TypeVar, Union, runtime_checkable

from .bootstrap import Operation, define, register


S, T = TypeVar("S"), TypeVar("T")


class Variable(Protocol[T]):
    __variable_name__: Hashable
    __variable_type__: Type[T]


@register(define(Variable))
class BaseVariable(Generic[T], Variable[T]):
    __variable_name__: Hashable
    __variable_type__: Type[T]

    def __init__(self, name: Hashable, type: Type[T]):
        self.__variable_name__ = name
        self.__variable_type__ = type

    def __repr__(self) -> str:
        return f"{self.__variable_name__}: {self.__variable_type__.__name__}"


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


class Environment(Protocol[T]):
    def __getitem__(self, key: Hashable) -> T: ...
    def __contains__(self, key: Hashable) -> bool: ...
    def keys(self) -> Iterable[Hashable]: ...


@register(define(Environment))
class BaseEnvironment(Generic[T], dict[Hashable, T]):
    pass


@define(Operation)
def head_of(term: Term[T]) -> Operation[T]:
    return term.__head__


@define(Operation)
def args_of(term: Term[T]) -> Iterable[Term[T] | T]:
    return term.__args__


@define(Operation)
def union(ctx: Environment[S], other: Environment[T]) -> Environment[S | T]:
    assert not set(ctx.keys()) & set(other.keys()), \
        "union only defined for disjoint contexts"
    return define(Environment)(
        [(k, ctx[k]) for k in ctx.keys()] + \
        [(k, other[k]) for k in other.keys()]
    )

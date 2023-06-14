from typing import Any, Callable, Container, ContextManager, Generic, Iterable, NamedTuple, Optional, Protocol, Set, TypeVar, Union, runtime_checkable

from .bootstrap import Operation, define


S, T = TypeVar("S"), TypeVar("T")


class Symbol(Generic[T]):
    pass


class Term(Protocol[T]):
    __head__: Operation[T]
    __args__: tuple["Term[T]" | T, ...]


@define(Operation)
def get_head(term: Term[T]) -> Operation[T]:
    raise TypeError(f"Expected term, got {term}")


@define(Operation)
def get_args(term: Term[T]) -> Iterable[Term[T] | T]:
    raise TypeError(f"Expected term, got {term}")


class Environment(Protocol[T]):
    pass


@define(Operation)
def read(ctx: Environment[T], key: Symbol[T]) -> T:
    ...


@define(Operation)
def union(ctx: Environment[S], other: Environment[T]) -> Environment[S | T]:
    ...


@define(Operation)
def keys(ctx: Environment[T]) -> Set[Symbol[T]]:
    ...


@define(Operation)
def write(ctx: Environment[T], key: Symbol[T], value: T) -> Environment[T]:
    return union(ctx, Environment((key, value)))


@define(Operation)
def contains(ctx: Environment[T], key: Symbol[T]) -> bool:
    return key in keys(ctx)

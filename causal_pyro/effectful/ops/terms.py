from typing import Any, Callable, Container, ContextManager, Generic, Iterable, NamedTuple, Optional, Protocol, Set, TypeVar, Union, runtime_checkable

import functools

from .bootstrap import define


S, T = TypeVar("S"), TypeVar("T")


class Symbol(Generic[T]):
    pass


class Operation(Protocol[T]):
    __symbol__: Symbol[T]
    __signature__: tuple[Symbol[T], ...]


@define(Operation)
def get_name(op: Operation[T]) -> Symbol[T]:
    if hasattr(op, "__symbol__"):
        return op.__symbol__
    raise TypeError(f"Expected operation, got {op}")


@define(Operation)
def get_signature(op: Operation[T]):
    if hasattr(op, "__signature__"):
        return op.__signature__
    raise TypeError(f"Expected operation, got {op}")


class Term(Protocol[T]):
    __head__: Operation[T]
    __args__: tuple["Term[T]" | T, ...]


@define(Operation)
def get_head(term: Term[T]) -> Operation[T]:
    if hasattr(term, "__head__"):
        return term.__head__
    raise TypeError(f"Expected term, got {term}")


@define(Operation)
def get_args(term: Term[T]) -> Iterable[Term[T] | T]:
    if hasattr(term, "__args__"):
        return term.__args__
    raise TypeError(f"Expected term, got {term}")


class Environment(Protocol[T]):
    pass


@define(Operation)
def read(ctx: Environment[T], key: Symbol[T]) -> T:
    ...


@define(Operation)
def write(ctx: Environment[T], key: Symbol[T], value: T) -> Environment[T]:
    ...


@define(Operation)
def keys(ctx: Environment[T]) -> Set[Symbol[T]]:
    ...


@define(Operation)
def contains(ctx: Environment[T], key: Symbol[T]) -> bool:
    return key in keys(ctx)


@define(Operation)
def union(ctx: Environment[S], other: Environment[T]) -> Environment[S | T]:
    result = ctx
    for key in keys(other):
        if contains(result, key):
            raise ValueError(f"Duplicate key: {key}")
        result = write(result, key, read(other, key))
    return result


class Computation(Protocol[T]):
    __ctx__: Environment[T]
    __value__: Term[T]


@define(Operation)
def get_ctx(obj: Computation[T]) -> Environment[T]:
    if hasattr(obj, "__ctx__"):
        return obj.__ctx__
    raise TypeError(f"Object {obj} has no context")


@define(Operation)
def get_value(obj: Computation[T]) -> Term[T]:
    if hasattr(obj, "__value__"):
        return obj.__value__
    raise TypeError(f"Object {obj} has no value")


class Interpretation(Generic[T], Environment[Operation[T]]):
    pass

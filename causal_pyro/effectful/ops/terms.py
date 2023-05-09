from typing import Any, Callable, Container, ContextManager, Generic, NamedTuple, Optional, Protocol, Set, TypeVar, Union, runtime_checkable

import functools

from .bootstrap import define, define_operation, define_meta


S, T = TypeVar("S"), TypeVar("T")


class Meta(Generic[T]):
    pass


define.register(Meta)(functools.partial(define_meta, Meta))
Meta = define(Meta)(Meta)


@define(Meta)
class Symbol(Generic[T]):
    pass


@define(Meta)
class Atom(Generic[T]):
    pass


@define(Meta)
class Operation(Generic[T]):
    __symbol__: Symbol[T]
    __signature__: tuple[Symbol[T], ...]


define.register(Operation)(functools.partial(define_operation, Operation))


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


@define(Meta)
class Term(Protocol[T]):
    __head__: Operation[T]
    __args__: tuple["Term[T]" | Atom[T], ...]


@define(Operation)
def get_head(term: Term[T]) -> Operation[T]:
    if hasattr(term, "__head__"):
        return term.__head__
    raise TypeError(f"Expected term, got {term}")


@define(Operation)
def get_args(term: Term[T]) -> tuple[Term[T] | Atom[T], ...]:
    if hasattr(term, "__args__"):
        return term.__args__
    raise TypeError(f"Expected term, got {term}")


@define(Meta)
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


@define(Meta)
class Object(Protocol[T]):
    __ctx__: Environment[T]
    __value__: Term[T]


@define(Operation)
def get_ctx(obj: Object[T]) -> Environment[T]:
    if hasattr(obj, "__ctx__"):
        return obj.__ctx__
    raise TypeError(f"Object {obj} has no context")


@define(Operation)
def get_value(obj: Object[T]) -> Term[T]:
    if hasattr(obj, "__value__"):
        return obj.__value__
    raise TypeError(f"Object {obj} has no value")

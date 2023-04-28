from typing import Any, Callable, Container, ContextManager, Generic, NamedTuple, Optional, Set, TypeVar, Union

from .bootstrap import define, define_operation, define_meta


S, T = TypeVar("S"), TypeVar("T")


class Meta(Generic[T]):
    pass


define.register(Meta)(define_meta)
Meta = define(Meta)(Meta)


@define(Meta)
class Symbol(Generic[T]):
    pass


@define(Meta)
class Atom(Generic[T]):
    pass


@define(Meta)
class Operation(Generic[T]):
    pass


define.register(Operation)(define_operation)


@define(Operation)
def get_name(op: Operation[T]) -> Symbol[T]:
    ...


@define(Operation)
def get_signature(op: Operation[T]):
    ...


@define(Meta)
class Term(Generic[T]):
    pass


@define(Operation)
def get_head(term: Term[T]) -> Operation[T]:
    ...


@define(Operation)
def get_args(term: Term[T]) -> tuple[Term[T], ...]:
    ...


@define(Meta)
class Environment(Generic[T]):
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
    ...


@define(Meta)
class Object(Generic[T]):
    pass


@define(Operation)
def get_ctx(obj: Object[T]) -> Environment[T]:
    ...


@define(Operation)
def get_value(obj: Object[T]) -> Term[T]:
    ...

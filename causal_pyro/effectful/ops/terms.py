from typing import Any, Callable, Container, ContextManager, Generic, NamedTuple, Optional, Set, TypeVar, Union

from ..internals.runtime import Kind, define, define_operation, define_form, define_meta, get_model


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


@define(Meta)
class Form(Generic[T]):
    pass


define.register(Form)(define_form)


@define(Operation)
def get_name(op: Operation[T] | Form[T]) -> Symbol[T]:
    ...


@define(Meta)
class Term(Generic[T]):
    pass


@define(Operation)
def get_head(term: Term[T]) -> Operation[T] | Form[T]:
    ...


@define(Operation)
def get_args(term: Term[T]) -> tuple[Term[T], ...]:
    ...


@define(Meta)
class Context(Generic[T]):
    pass


@define(Operation)
def read(ctx: Context[T], key: Symbol[T]) -> T:
    ...


@define(Operation)
def write(ctx: Context[T], key: Symbol[T], value: T) -> Context[T]:
    ...


@define(Operation)
def keys(ctx: Context[T]) -> Set[Symbol[T]]:
    ...


@define(Operation)
def contains(ctx: Context[T], key: Symbol[T]) -> bool:
    return key in keys(ctx)


@define(Operation)
def union(ctx: Context[S], other: Context[T]) -> Context[S | T]:
    ...


@define(Meta)
class Object(Generic[T]):
    pass


@define(Operation)
def get_ctx(obj: Object[T]) -> Context[T]:
    ...


@define(Operation)
def get_value(obj: Object[T]) -> Term[T]:
    ...

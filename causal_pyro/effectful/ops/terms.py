from typing import Any, Callable, Container, ContextManager, Generic, NamedTuple, Optional, Type, TypeVar, Union

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


@define(Operation)
def get_name(op: Operation[T]) -> Symbol[T]:
    ...


@define(Meta)
class Form(Generic[T]):
    pass


define.register(Form)(define_form)


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
def contains(ctx: Context[T], key: Symbol[T]) -> bool:
    ...


@define(Operation)
def union(ctx: Context[T], other: Context[S]) -> Context[S | T]:
    ...


@define(Operation)
def substitute(term: Term[T], ctx: Context[T]) -> Term[T]:
    ...


@define(Operation)
def fvs(term: Term[T]) -> Context[Type[T]]:
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

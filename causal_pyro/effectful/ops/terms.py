from typing import Any, Callable, Container, ContextManager, Generic, NamedTuple, Optional, Type, TypeVar, Union

from ..internals.runtime import Kind, define, define_operation, define_form, define_kind, get_model


S, T = TypeVar("S"), TypeVar("T")


class Kind(Generic[T]):
    pass


define.register(Kind)(define_kind)
Kind = define(Kind)(Kind)


@define(Kind)
class Symbol(Generic[T]):
    pass


@define(Kind)
class Atom(Generic[T]):
    pass


@define(Kind)
class Operation(Generic[T]):
    pass


define.register(Operation)(define_operation)


@define(Kind)
class Form(Generic[T]):
    pass


define.register(Form)(define_form)


@define(Kind)
class Term(Generic[T]):
    pass


@define(Operation)
def get_head(term: Term[T]) -> Operation[T] | Form[T]:
    ...


@define(Operation)
def get_args(term: Term[T]) -> tuple[Term[T], ...]:
    ...


@define(Kind)
class Context(Generic[T]):
    pass


@define(Operation)
def substitute(term: Term[T], ctx: Context[T]) -> Term[T]:
    ...


@define(Operation)
def fvs(term: Term[T]) -> Context[None]:
    ...


@define(Kind)
class Object(Generic[T]):
    pass


@define(Operation)
def get_ctx(obj: Object[T]) -> Context[T]:
    ...


@define(Operation)
def get_value(obj: Object[T]) -> Term[T]:
    ...

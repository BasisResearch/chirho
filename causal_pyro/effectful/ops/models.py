from typing import Generic, List, Optional, TypeVar

from .terms import Kind, Operation, Term, Context, Form, define, get_head, get_args


S, T = TypeVar("S"), TypeVar("T")


@define(Kind)
class Model(Generic[T]):
    pass

# Model ?= Context[Term[T]]


@define(Operation)
def union(model: Model[T], other: Model[T]) -> Model[T]:
    ...


@define(Operation)
def compose(model: Model[T], other: Model[T]) -> Model[T]:
    ...


@define(Operation)
def product(model: Model[T], comodel: Model[T]) -> Model[T]:
    ...


@define(Operation)
def quotient(model: Model[T], other: Model[T]) -> Model[T]:
    ...

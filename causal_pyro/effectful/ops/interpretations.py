from typing import Generic, List, Optional, TypeVar

from .terms import Meta, Operation, Term, Context, Form, define, get_name, get_head, get_args, read


S, T = TypeVar("S"), TypeVar("T")


@define(Meta)
class Interpretation(Generic[T]):
    pass

# Model ?= Context[Term[T]]

@define(Operation)
def get_model(model: Interpretation[T], op: Operation[T]) -> Term[T]:
    return read(model, get_name(op))


@define(Operation)
def union(model: Interpretation[T], other: Interpretation[T]) -> Interpretation[T]:
    ...


@define(Operation)
def compose(model: Interpretation[T], other: Interpretation[T]) -> Interpretation[T]:
    ...


@define(Operation)
def product(model: Interpretation[T], comodel: Interpretation[T]) -> Interpretation[T]:
    ...


@define(Operation)
def quotient(model: Interpretation[T], other: Interpretation[T]) -> Interpretation[T]:
    ...

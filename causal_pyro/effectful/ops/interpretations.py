from typing import Generic, List, Optional, TypeVar

from causal_pyro.effectful.ops.bootstrap import define
from causal_pyro.effectful.ops.interpreter import T
from causal_pyro.effectful.ops.terms import T, Environment, Operation, define

from .terms import Meta, Operation, Term, Environment, define, get_name, read


S, T = TypeVar("S"), TypeVar("T")


@define(Meta)
class Interpretation(Generic[T], Environment[Operation[T]]):
    pass


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


@define(Operation)
def cont(ctx: Environment[T], result: Optional[T]) -> T:
    ...


@define(Operation)
def reflect(ctx: Environment[T], result: Optional[T]) -> T:
    ...

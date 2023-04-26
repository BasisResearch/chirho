from typing import Generic, List, Optional, TypeVar

from .terms import Kind, Operation, Term, Context, Form, define, get_head, get_args


S, T = TypeVar("S"), TypeVar("T")


@define(Kind)
class Model(Generic[T]):
    pass


@define(Operation)
def evaluate(ctx: Context[T], model: Model[T], term: Term[T]) -> T:
    # inr(product(model, comodel)) == comodel
    # inl(product(model, comodel)) == model
    # sem = get_model(inl(model), get_head(term))(ctx, *get_args(term))
    # return evaluate(ctx, inr(model), sem)
    ...


@define(Form)
def apply(ctx: Context[T], model: Model[T], op: Operation[T], *args: Term[T]) -> T:
    # return get_model(model, op)(*(evaluate(ctx, model, arg) for arg in args))
    ...


################################################################################


@define(Operation)
def union(model: Model[T], other: Model[T]) -> Model[T]:
    ...


@define(Operation)
def compose(model: Model[T], other: Model[T]) -> Model[T]:
    ...


@define(Operation)
def product(model: Model[T], comodel: Model[T]) -> Model[T]:
    ...


################################################################################


@define(Operation)
def quotient(model: Model[T], other: Model[T]) -> Model[T]:
    ...


################################################################################


@define(Operation)
def cont(ctx: Context[T], result: Optional[T]) -> T:
    ...


@define(Operation)
def reflect(ctx: Context[T], result: Optional[T]) -> T:
    ...

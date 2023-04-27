from typing import Generic, List, Optional, TypeVar

from .terms import Kind, Operation, Term, Context, Form, define, get_head, get_args


S, T = TypeVar("S"), TypeVar("T")


@define(Form)
def evaluate(ctx: Context[T], model: Model[T], term: Term[T]) -> T:
    # inr(product(model, comodel)) == comodel
    # inl(product(model, comodel)) == model
    # sem = get_model(inl(model), get_head(term))(ctx, *get_args(term))
    # return evaluate(ctx, inr(model), sem)
    return get_model(model, apply)(ctx, model, get_head(term), *get_args(term))


@define(Form)
def apply(ctx: Context[T], model: Model[T], op: Operation[T], *args: Term[T]) -> T:
    return get_model(model, op)(*(evaluate(ctx, model, arg) for arg in args))


@define(Operation)
def cont(ctx: Context[T], result: Optional[T]) -> T:
    ...


@define(Operation)
def reflect(ctx: Context[T], result: Optional[T]) -> T:
    ...

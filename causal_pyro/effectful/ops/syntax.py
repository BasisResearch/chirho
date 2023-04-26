from typing import TypeVar

from .models import Model, evaluate, apply
from .terms import Form, Context, Term, define


S, T = TypeVar("S"), TypeVar("T")


@define(Form)
def Return(ctx: Context[T], model: Model[T], value: Term[T]) -> T:
    return evaluate(ctx, model, value)


@define(Form)
def cond(ctx: Context[S | T], model: Model[S | T], test: bool, body: Term[S], orelse: Term[T]) -> S | T:
    return evaluate(ctx, model, body) if test else evaluate(ctx, model, orelse)

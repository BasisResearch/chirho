from typing import TypeVar

from ..ops.interpretations import Interpretation
from ..ops.interpreter import evaluate
from ..ops.terms import Form, Environment, Symbol, Term, define, union


S, T = TypeVar("S"), TypeVar("T")


@define(Form)
def Return(ctx: Environment[T], model: Interpretation[T], value: Term[T]) -> T:
    return evaluate(ctx, model, value)


@define(Form)
def Cond(ctx: Environment[S | T], model: Interpretation[S | T], test: bool, body: Term[S], orelse: Term[T]) -> S | T:
    return evaluate(ctx, model, body) if test else evaluate(ctx, model, orelse)


@define(Form)
def Let(ctx: Environment[T], model: Interpretation[T], name: Symbol[T], value: Term[T], body: Term[T]) -> T:
    value = evaluate(ctx, model, value)
    return evaluate(union(ctx, {name: value}), model, body)


@define(Form)
def Begin(ctx: Environment[T], model: Interpretation[T], *terms: Term[T]) -> T:
    for term in terms[:-1]:
        evaluate(ctx, model, term)
    return evaluate(ctx, model, terms[-1])

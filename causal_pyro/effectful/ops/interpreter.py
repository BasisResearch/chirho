from typing import Generic, List, Type, TypeVar

from causal_pyro.effectful.internals.runtime import define
from causal_pyro.effectful.ops.terms import T, Context, Operation, Term

from .terms import Meta, Operation, Term, Context, Form, define, get_head, get_args, get_ctx
from .interpretations import Interpretation, get_name, read


S, T = TypeVar("S"), TypeVar("T")


@define(Form)
def evaluate(ctx: Context[T], interpretation: Interpretation[T], term: Term[T]) -> T:
    return read(interpretation, get_name(apply))(ctx, interpretation, get_head(term), *get_args(term))


@define(Form)
def apply(ctx: Context[T], interpretation: Interpretation[T], op: Operation[T], *args: Term[T]) -> T:
    return read(interpretation, get_name(op))(*(evaluate(ctx, interpretation, arg) for arg in args))


@define(Form)
def typeof(judgements: Interpretation[Type[T]], term: Term[T]) -> Type[T]:
    return evaluate(ctx, judgements, term)


@define(Form)
def fvs(judgements: Interpretation[Type[T]], term: Term[T]) -> Context[Type[T]]:
    return get_ctx(typeof(judgements, term))


@define(Form)
def substitute(term: Term[T], ctx: Context[T]) -> Term[T]:
    return evaluate(ctx, default, term)

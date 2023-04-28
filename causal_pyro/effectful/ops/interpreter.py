from typing import Generic, List, Optional, Type, TypeVar

from causal_pyro.effectful.internals.runtime import define
from causal_pyro.effectful.ops.terms import T, Context, Operation, Term

from .terms import Meta, Operation, Term, Context, Form, define, get_head, get_args
from .interpretations import Interpretation, get_name, read


S, T = TypeVar("S"), TypeVar("T")


@define(Form)
def evaluate(ctx: Context[T], interpretation: Interpretation[T], term: Term[T]) -> T:
    return read(interpretation, get_name(apply))(ctx, interpretation, get_head(term), *get_args(term))


@define(Form)
def apply(ctx: Context[T], interpretation: Interpretation[T], op: Operation[T], *args: Term[T]) -> T:
    return read(interpretation, get_name(op))(*(evaluate(ctx, interpretation, arg) for arg in args))


@define(Operation)
def cont(ctx: Context[T], result: Optional[T]) -> T:
    ...


@define(Operation)
def reflect(ctx: Context[T], result: Optional[T]) -> T:
    ...


@define(Form)
def typeof(term: Term[T]) -> Type[T]:
    ...


@define(Form)
def fvs(term: Term[T]) -> Context[Type[T]]:
    ...


@define(Form)
def substitute(term: Term[T], ctx: Context[T]) -> Term[T]:
    ...

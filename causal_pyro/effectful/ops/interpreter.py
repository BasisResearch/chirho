from typing import Generic, List, Type, TypeVar

from causal_pyro.effectful.ops.bootstrap import define
from causal_pyro.effectful.ops.terms import T, Environment, Operation, Term

from .terms import Computation, Operation, Symbol, Term, Environment, Operation, define, get_head, get_args, get_ctx, get_value
from .interpretations import Interpretation, get_name, read


S, T = TypeVar("S"), TypeVar("T")


@define(Operation)
def traverse(interpretation: Interpretation[S], obj: Computation[T]) -> Computation[S]:
    """
    Generic meta-circular transformation of a term in a context.
    Use for evaluation, substitution, typechecking, etc.
    """
    ctx: Environment[T] = get_ctx(obj)
    term: Term[T] = get_value(obj)
    return read(interpretation, get_name(apply))(
        interpretation,
        get_head(term),
        *(Computation(ctx, arg) for arg in get_args(term))
    )


@define(Operation)
def t_apply(interpretation: Interpretation[S], op: Operation[T], *args: Computation[T]) -> Computation[S]:
    ctx = sum(*(get_ctx(arg) for arg in args), Environment())
    return read(interpretation, get_name(op))(
        *(traverse(interpretation, arg) for arg in args)
    )


@define(Operation)
def evaluate(ctx: Environment[T], interpretation: Interpretation[T], term: Term[T]) -> T:
    return read(interpretation, get_name(apply))(ctx, interpretation, get_head(term), *get_args(term))


@define(Operation)
def apply(ctx: Environment[T], interpretation: Interpretation[T], op: Operation[T], *args: Term[T]) -> T:
    return read(interpretation, get_name(op))(*(evaluate(ctx, interpretation, arg) for arg in args))


@define(Operation)
def typeof(ctx: Environment[Type[T]], judgements: Interpretation[Type[T]], term: Term[T]) -> Type[T]:
    return evaluate(ctx, judgements, term)


@define(Operation)
def fvs(judgements: Interpretation[Type[T]], term: Term[T]) -> Environment[Type[T]]:
    return get_ctx(typeof(judgements, term))


@define(Operation)
def substitute(term: Term[T], ctx: Environment[T]) -> Term[T]:
    return evaluate(ctx, default, term)


@define(Operation)
def rename(term: Term[T], ctx: Environment[Symbol[T]]) -> Term[T]:
    return substitute(term, ctx)


@define(Operation)
def pprint(reprs: Interpretation[str], term: Term[T]) -> str:
    return evaluate({}, reprs, term)
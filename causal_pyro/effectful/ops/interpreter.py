from typing import Generic, List, Type, TypeVar

from .terms import Computation, Operation, Symbol, Term, Environment, Operation, define, get_head, get_args, get_ctx, get_value
from .interpretations import Interpretation, get_name, read


S, T = TypeVar("S"), TypeVar("T")


# @define(Operation)  # TODO self-hosting
def apply(interpretation: Interpretation[S], op: Operation[T], *args: Computation[T]) -> Computation[S]:
    ctx = sum(*(get_ctx(arg) for arg in args), Environment())
    return read(interpretation, get_name(op))(
        *(traverse(interpretation, arg) for arg in args)
    )


# @define(Operation)  # TODO self-hosting
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


#########################################################
# Special built-in interpretations
#########################################################


@define(Operation)
def evaluate(ctx: Environment[T], interpretation: Interpretation[T], term: Term[T]) -> T:
    return traverse(interpretation, Computation(ctx, term))


@define(Operation)
def typeof(ctx: Environment[Type[T]], judgements: Interpretation[Type[T]], term: Term[T]) -> Type[T]:
    return traverse(judgements, Computation(ctx, term))


@define(Operation)
def fvs(judgements: Interpretation[Type[T]], term: Term[T]) -> Environment[Type[T]]:
    return get_ctx(typeof(judgements, term))


@define(Operation)
def substitute(term: Term[T], ctx: Environment[T]) -> Term[T]:
    return traverse((), Computation(ctx, term))


@define(Operation)
def rename(term: Term[T], ctx: Environment[Symbol[T]]) -> Term[T]:
    return substitute(term, ctx)


@define(Operation)
def pprint(reprs: Interpretation[str], term: Term[T]) -> str:
    return traverse(reprs, term)

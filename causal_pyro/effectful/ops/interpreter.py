from typing import Generic, List, Protocol, Type, TypeVar

from .terms import Operation, Symbol, Term, Environment, Operation, define, get_head, get_args, get_ctx, get_value
from .interpretations import Interpretation, get_name, read


S, T = TypeVar("S"), TypeVar("T")


class Computation(Protocol[T]):
    __ctx__: Environment[T]
    __value__: Term[T]


@define(Operation)
def get_ctx(obj: Computation[T]) -> Environment[T]:
    if hasattr(obj, "__ctx__"):
        return obj.__ctx__
    raise TypeError(f"Object {obj} has no context")


@define(Operation)
def get_value(obj: Computation[T]) -> Term[T]:
    if hasattr(obj, "__value__"):
        return obj.__value__
    raise TypeError(f"Object {obj} has no value")


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

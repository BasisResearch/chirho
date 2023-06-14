from typing import Generic, List, Protocol, Type, TypeVar

from causal_pyro.effectful.ops.bootstrap import Interpretation, Operation, define, get_interpretation, register
from causal_pyro.effectful.ops.interpretations import fwd, handler, reflect, runner
from causal_pyro.effectful.ops.terms import Term, Environment, Operation, Variable, head_of, args_of, union


S, T = TypeVar("S"), TypeVar("T")


class Computation(Protocol[T]):
    __ctx__: Environment[T]
    __value__: Term[T]


@register(None, define(Computation))
class BaseComputation(Generic[T], Computation[T]):
    __ctx__: Environment[T]
    __value__: Term[T]

    def __init__(self, ctx: Environment[T], value: Term[T]):
        self.__ctx__ = ctx
        self.__value__ = value

    def __repr__(self) -> str:
        return f"{self.__value__} @ {self.__ctx__}"


@define(Operation)
def ctx_of(obj: Computation[T]) -> Environment[T]:
    if hasattr(obj, "__ctx__"):
        return obj.__ctx__
    raise TypeError(f"Object {obj} has no context")


@define(Operation)
def value_of(obj: Computation[T]) -> Term[T]:
    if hasattr(obj, "__value__"):
        return obj.__value__
    raise TypeError(f"Object {obj} has no value")


@define(Operation)
def apply(intp: Interpretation[S], op: Operation[T], *args: Computation[T]) -> Computation[S]:
    return intp[op](*(traverse(intp, arg) for arg in args))


@define(Operation)
def traverse(intp: Interpretation[S], obj: Computation[T]) -> Computation[S]:
    """
    Generic meta-circular transformation of a term in a context.
    Use for evaluation, substitution, typechecking, etc.
    """
    ctx, term = ctx_of(obj), value_of(obj)
    op, args = head_of(term), args_of(term)
    return apply(intp, op, *(define(Computation)(ctx, arg) for arg in args))


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
def rename(term: Term[T], ctx: Environment[Variable[T]]) -> Term[T]:
    return substitute(term, ctx)


@define(Operation)
def pprint(reprs: Interpretation[str], term: Term[T]) -> str:
    return traverse(reprs, term)

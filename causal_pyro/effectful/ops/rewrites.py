from typing import Callable, Generic, Hashable, Iterable, List, Optional, Protocol, Type, TypeVar

import functools

from causal_pyro.effectful.ops.environments import Environment, union
from causal_pyro.effectful.ops.operations import Interpretation, Operation, define, register
from causal_pyro.effectful.ops.interpretations import product, reflect
from causal_pyro.effectful.ops.terms import Term, Variable, LazyInterpretation, head_of, args_of


S = TypeVar("S")
T = TypeVar("T")


class Computation(Protocol[T]):
    __ctx__: Environment[T]
    __value__: T


@register(define(Computation))
class _BaseComputation(Generic[T], Computation[T]):
    __ctx__: Environment[T]
    __value__: T

    def __init__(self, ctx: Environment[T], value: T):
        self.__ctx__ = ctx
        self.__value__ = value

    def __repr__(self) -> str:
        return f"{self.__value__} @ {self.__ctx__}"


@define(Operation)
def ctx_of(obj: Computation[T]) -> Environment[T]:
    return obj.__ctx__


@define(Operation)
def value_of(obj: Computation[T]) -> T:
    return obj.__value__


def ContextualInterpretation(intp: Interpretation[T]) -> Interpretation[Computation[T]]:

    def apply(
        op_intp: Callable[..., T], res: Optional[S], *args: Computation[T], **kwargs
    ) -> Computation[T]:
        ctx: Environment[T] = union(*(ctx_of(arg) for arg in args))
        value: T = op_intp(res, *(value_of(arg) for arg in args), **kwargs)
        return define(Computation)(ctx, value)

    return define(Interpretation)({
        op: functools.partial(apply, intp[op]) for op in intp.keys()
    })


LazyComputation = Computation[Term[T]]


@define(Operation)
def match(op_intp: Callable, res: Optional[T], *args: T | Term[T] | Variable[T], **kwargs) -> bool:
    return res is not None or len(args) == 0 or \
        any(isinstance(arg, (Variable, Term)) for arg in args)


@define(Operation)
def traverse(obj: LazyComputation[T]) -> LazyComputation[S]:
    """
    Generic meta-circular transformation of a term in a context.
    Use for evaluation, substitution, typechecking, etc.
    """
    ctx, op, args = ctx_of(obj), head_of(value_of(obj)), args_of(value_of(obj))
    return op(*(define(Computation)(ctx, arg) for arg in args))


def MetacircularInterpretation(intp: Interpretation[T]) -> Interpretation[LazyComputation[T]]:

    def apply(
        op_intp: Callable[..., S], res: Optional[T], *args: LazyComputation[T], **kwargs
    ) -> LazyComputation[S]:

        args_: tuple[LazyComputation[S], ...] = tuple(traverse(arg) for arg in args)

        ctx: Environment[S | Term[S]] = union(*(ctx_of(arg) for arg in args_))

        value = op_intp(res, *(value_of(arg) for arg in args_), **kwargs) \
            if match(op_intp, res, *(value_of(arg) for arg in args_), **kwargs) \
            else reflect(res)

        return define(Computation)(ctx, value)

    return product(LazyInterpretation(*intp.keys()), define(Interpretation)({
        op: functools.partial(apply, intp[op]) for op in intp.keys()
    }))


#########################################################
# Special built-in interpretations
#########################################################

# @define(Operation)
# def evaluate(interpretation: Interpretation[T], term: Term[T], ctx: Optional[Environment[T]] = None) -> T:
#     if ctx is None:
#         ctx = define(Environment)()
#     return interpreter(interpretation)(traverse(define(Computation)(ctx, term)))
#
#
# @define(Operation)
# def typeof(judgements: Interpretation[Type[T]], term: Term[T], ctx: Optional[Environment[Type[T]]] = None) -> Type[T]:
#     if ctx is None:
#         ctx = define(Environment)()
#     return traverse(judgements, define(Computation)(ctx, term))
#
#
# @define(Operation)
# def fvs(judgements: Interpretation[Type[T]], term: Term[T]) -> Environment[Type[T]]:
#     return ctx_of(typeof(judgements, term))
#
#
# @define(Operation)
# def substitute(term: Term[T], ctx: Environment[T]) -> Term[T]:
#     return traverse(define(Computation)(ctx, term))
#
#
# @define(Operation)
# def rename(term: Term[T], ctx: Environment[Variable[T]]) -> Term[T]:
#     return substitute(term, ctx)
#
#
# @define(Operation)
# def pprint(reprs: Interpretation[str], term: Term[T]) -> str:
#     return interpreter(reprs)(traverse)(term)

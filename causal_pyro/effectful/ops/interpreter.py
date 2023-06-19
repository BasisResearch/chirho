from typing import Callable, Generic, Hashable, Iterable, List, Optional, Protocol, Type, TypeVar

import functools

from causal_pyro.effectful.ops.bootstrap import Interpretation, Operation, define, register
from causal_pyro.effectful.ops.interpretations import fwd, handler, interpreter, product, reflect, reflections, runner
from causal_pyro.effectful.ops.terms import Term, Variable, LazyInterpretation, head_of, args_of


S = TypeVar("S")
T = TypeVar("T")


class Environment(Protocol[T]):
    def __getitem__(self, key: Hashable) -> T: ...
    def __contains__(self, key: Hashable) -> bool: ...
    def keys(self) -> Iterable[Hashable]: ...


@register(define(Environment))
class BaseEnvironment(Generic[T], dict[Hashable, T]):
    pass


@define(Operation)
def union(ctx: Environment[S], other: Environment[T]) -> Environment[S | T]:
    assert not set(ctx.keys()) & set(other.keys()), \
        "union only defined for disjoint contexts"
    return define(Environment)(
        [(k, ctx[k]) for k in ctx.keys()] + \
        [(k, other[k]) for k in other.keys()]
    )


class Computation(Protocol[T]):
    __ctx__: Environment[T]
    __value__: Term[T]


@register(define(Computation))
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
    return obj.__ctx__


@define(Operation)
def value_of(obj: Computation[T]) -> Term[T]:
    return obj.__value__


###########################################################

@define(Operation)
def match(op_intp: Callable, res: Optional[T], *args: T | Term[T] | Variable[T], **kwargs) -> bool:
    return res is not None or len(args) == 0 or \
        any(isinstance(arg, (Variable, Term)) for arg in args)


@define(Operation)
def traverse(obj: Computation[T]) -> Computation[S]:
    """
    Generic meta-circular transformation of a term in a context.
    Use for evaluation, substitution, typechecking, etc.
    """
    ctx, op, args = ctx_of(obj), head_of(value_of(obj)), args_of(value_of(obj))
    return op(*(define(Computation)(ctx, arg) for arg in args))


@define(Operation)
def MetacircularInterpretation(intp: Interpretation[T]) -> Interpretation[Computation[T]]:

    def apply(
        op_intp: Callable[..., S], res: Optional[T], *args: Computation[T], **kwargs
    ) -> Computation[S]:
        args_: Computation[S] = tuple(traverse(arg) for arg in args)
        ctx: Environment[S] = union(*(ctx_of(arg) for arg in args_))
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

from typing import Callable, Generic, List, Optional, Protocol, Type, TypeVar

import functools

from causal_pyro.effectful.ops.bootstrap import Interpretation, Operation, define
from causal_pyro.effectful.ops.interpretations import fwd, handler, interpreter, product, reflect, reflections, runner
from causal_pyro.effectful.ops.terms import Term, Environment, Variable, Computation, \
    ctx_of, value_of, head_of, args_of, union


S = TypeVar("S")
T = TypeVar("T")


LazyVal = T | Term[T] | Variable[T]

@define(Operation)
def LazyInterpretation(*ops: Operation[T]) -> Interpretation[LazyVal[T]]:
    return product(reflections(define(Term)), define(Interpretation)({
        op: functools.partial(define(Term), op) for op in ops
    }))


@define(Operation)
def match(op_intp: Callable, res: Optional[T], *args: LazyVal[T], **kwargs) -> bool:
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

import contextlib
import functools
from typing import Callable, Concatenate, Mapping, Optional, ParamSpec, Tuple, TypeVar

from chirho.meta.ops.core import Interpretation, Operation, Term, Variable, define, evaluate
from chirho.meta.ops.interpreter import interpreter

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@define(Operation)
def quotient(
    intp: Interpretation[S, T],
    *intps: Interpretation[S, T]
) -> Interpretation[S, T]:
    if len(intps) == 0:
        return intp
    if len(intps) > 1:
        raise NotImplementedError("Associativity laws not yet clear")

    intp2, = intps

    def _wrapper(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        with interpreter(intp2):
            new_args = tuple(evaluate(arg) for arg in args)
            new_kwargs = {k: evaluate(kwarg) for k, kwarg in kwargs.items()}
        return fn(*new_args, **new_kwargs)

    return {op: functools.partial(_wrapper, intp[op]) for op in intp.keys()}


@define(Operation)
def specializer(intp: Interpretation[S, T]) -> Interpretation[S, Term[T]]:

    def _free_wrapper(op, *args, **kwargs) -> Term[S]:
        return define(Term)(op, args, kwargs)

    def _partial_wrapper(op, fn, *args, **kwargs) -> Term[T]:
        if any(isinstance(arg, Variable) for arg in tuple(args) + tuple(kwargs.items())):
            return define(Term)(op, args, kwargs)
        return fn(*args, **kwargs)

    free = {op: functools.partial(_free_wrapper, op) for op in intp.keys()}
    partial = {op: functools.partial(_partial_wrapper, op, intp[op]) for op in intp.keys()}
    return quotient(free, partial)


@contextlib.contextmanager
def canonicalizer(intp: Interpretation[S, Term[T]]):
    from ..internals.runtime import get_interpretation

    curr_intp = get_interpretation()
    with interpreter(quotient(curr_intp, intp)):
        yield intp

import contextlib
import functools
from typing import Callable, ParamSpec, TypeVar

from chirho.meta.ops.core import Interpretation, Operation, Term, define, evaluate
from chirho.meta.ops.interpreter import interpreter

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@define(Operation)
def free_interpretation(*ops: Operation[P, T]) -> Interpretation[T, Term[T]]:

    def _free_wrapper(op: Operation[P, T], *args: P.args, **kwargs: P.kwargs) -> Term[T]:
        return define(Term)(op, args, kwargs)

    return {op: functools.partial(_free_wrapper, op) for op in ops}


@define(Operation)
def quotient(
    intp: Interpretation[S, Term[T]],
    *intps: Interpretation[S, Term[T]],
    # TODO should this have NbE-ish prompts for reification and reflection?
) -> Interpretation[S, Term[T]]:
    if len(intps) == 0:
        return intp
    if len(intps) > 1:
        raise NotImplementedError("Associativity laws not yet clear")

    intp2, = intps

    def _wrapper(fn: Callable[P, Term[T]], *args: P.args, **kwargs: P.kwargs) -> Term[T]:
        with interpreter(intp2):
            simplified_args = tuple(evaluate(arg) for arg in args)
            simplified_kwargs = {k: evaluate(kwarg) for k, kwarg in kwargs.items()}
        return fn(*simplified_args, **simplified_kwargs)

    return {op: functools.partial(_wrapper, intp[op]) for op in intp.keys()}


@contextlib.contextmanager
def simplifier(intp: Interpretation[S, Term[T]]):
    from ..internals.runtime import get_interpretation

    with interpreter(quotient(get_interpretation(), intp)):
        yield intp

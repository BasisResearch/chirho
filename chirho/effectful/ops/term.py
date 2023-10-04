from typing import Iterable, Mapping, ParamSpec, Protocol, TypeVar

import functools
import typing

from chirho.effectful.ops.interpretation import Interpretation, register
from chirho.effectful.ops.operation import Operation, define

from ..internals.base_term import _BaseTerm

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Term(Protocol[T]):
    op: Operation[..., T]
    args: Iterable["Term[T]" | T]
    kwargs: Mapping[str, "Term[T]" | T]


@define(Operation)
def evaluate(term: Term[T]) -> T:
    return term.op(
        *(evaluate(a) if isinstance(a, Term) else a for a in term.args),
        **{k: (evaluate(v) if isinstance(v, Term) else v) for k, v in term.kwargs.items()}
    )


def LazyInterpretation(*ops: Operation[P, T]) -> Interpretation[T, Term[T]]:
    return {
        op: functools.partial(lambda op, _, *args, **kwargs: define(Term)(op, args, kwargs), op)
        for op in ops
    }


register(define(Term), None, _BaseTerm)

from typing import Iterable, Mapping, ParamSpec, Protocol, TypeVar

import typing

from chirho.effectful.ops.interpretation import Interpretation, register
from chirho.effectful.ops.operation import Operation, define

from ..internals.base_term import _BaseTerm

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@typing.runtime_checkable
class Term(Protocol[T]):
    __op__: Operation[..., T]
    __args__: Iterable["Term[T]" | T]
    __kwargs__: Mapping[str, "Term[T]" | T]


@define(Operation)
def evaluate(term: Term[T]) -> T:
    return term.__op__(
        *(evaluate(a) if isinstance(a, Term) else a for a in term.__args__),
        **{k: (evaluate(v) if isinstance(v, Term) else v) for k, v in term.__kwargs__.items()}
    )


@define(Operation)
def LazyInterpretation(*ops: Operation[P, T]) -> Interpretation[T | Term[T]]:
    return define(Interpretation)({
        op: lambda _, *args, **kwargs: define(Term)(op, args, kwargs)
        for op in ops
    })


register(define(Term), None, _BaseTerm)

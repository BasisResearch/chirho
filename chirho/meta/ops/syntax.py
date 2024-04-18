import collections.abc
import functools
import typing
from typing import Callable, Iterable, Mapping, Optional, ParamSpec, Protocol, Type, TypeGuard, TypeVar

from ..internals.utils import weak_memoize

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)


@typing.runtime_checkable
class Operation(Protocol[P, T_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        ...

    def default(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        ...


@typing.runtime_checkable
class Term(Protocol[T]):
    op: Operation[..., T]
    args: Iterable["Term[T]" | T]
    kwargs: Mapping[str, "Term[T]" | T]


Interpretation = Mapping[Operation[..., T], Callable[..., V]]


def define(m: Type[T] | Callable[Q, T]) -> Operation[..., T]:
    """
    Scott encoding of a type as its constructor.
    """
    from ..internals.bootstrap import base_define

    return base_define(m)


@typing.overload
def register(op: Operation[P, T]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    ...


@typing.overload
def register(
    op: Operation[P, T],
    intp: Optional[Interpretation[T, V]],
) -> Callable[[Callable[Q, V]], Callable[Q, V]]:
    ...


@typing.overload
def register(
    op: Operation[P, T],
    intp: Optional[Interpretation[T, V]],
    interpret_op: Callable[Q, V],
) -> Callable[Q, V]:
    ...


def register(op, intp=None, interpret_op=None):
    if interpret_op is None:
        return lambda interpret_op: register(op, intp, interpret_op)

    if intp is None:
        setattr(op, "default", interpret_op)
        return interpret_op
    elif isinstance(intp, collections.abc.MutableMapping):
        intp.__setitem__(op, interpret_op)
        return interpret_op
    raise NotImplementedError(f"Cannot register {op} in {intp}")


def LazyInterpretation(*ops: Operation[P, T]) -> Interpretation[T, Term[T]]:
    return {
        op: functools.partial(
            lambda op, *args, **kwargs: define(Term)(op, args, kwargs), op
        )
        for op in ops
    }


def apply(
    interpretation: Interpretation[S, T],
    op: Operation[P, S],
    *args: P.args,
    **kwargs: P.kwargs
) -> T:
    try:
        interpret = interpretation[op]
    except KeyError:
        interpret = op.default
    return interpret(*args, **kwargs)


@define(Operation)
def evaluate(term: Term[T]) -> T:
    return term.op(
        *(evaluate(a) if isinstance(a, Term) else a for a in term.args),
        **{
            k: (evaluate(v) if isinstance(v, Term) else v)
            for k, v in term.kwargs.items()
        }
    )

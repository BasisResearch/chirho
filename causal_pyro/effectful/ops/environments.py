from typing import Callable, Generic, Hashable, Iterable, List, Optional, Protocol, Type, TypeVar

import functools

from causal_pyro.effectful.ops.operations import Interpretation, Operation, define, register


S = TypeVar("S")
T = TypeVar("T")


class Environment(Protocol[T]):
    def __getitem__(self, key: Hashable) -> T: ...
    def __contains__(self, key: Hashable) -> bool: ...
    def keys(self) -> Iterable[Hashable]: ...


@register(define(Environment))
class _BaseEnvironment(Generic[T], dict[Hashable, T]):
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

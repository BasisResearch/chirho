from typing import Any, Callable, Generic, Hashable, Iterable, Optional, Protocol, Type, TypeVar

import functools
import typing

from causal_pyro.effectful.ops.operations import Interpretation, Operation, define, register


S = TypeVar("S")
T = TypeVar("T")


class Variable(Protocol[T]):
    __variable_name__: str
    __variable_type__: Optional[Type[T]]


@register(define(Variable))
class _BaseVariable(Generic[T], Variable[T]):
    __variable_name__: str
    __variable_type__: Optional[Type[T]]

    def __init__(self, name: str, type: Optional[Type[T]] = None):
        self.__variable_name__ = name
        self.__variable_type__ = type

    def __repr__(self) -> str:
        return f"{self.__variable_name__}: {getattr(self, '__variable_type___', None)}"


@typing.runtime_checkable
class Environment(Protocol[T]):
    def __getitem__(self, var: Variable[T]) -> T: ...
    def __setitem__(self, var: Variable[T], value: T) -> None: ...
    def __contains__(self, var: Variable[T]) -> bool: ...
    def keys(self) -> Iterable[Hashable]: ...


@register(define(Environment))
class _BaseEnvironment(Generic[T], dict[Variable[T], T | Variable[T] | Environment[T]]):

    @define(Operation)
    def __getitem__(self, var: Variable[T]) -> T | Variable[T] | Environment[T]:
        if var not in self:
            raise ValueError(f"variable {var} not defined")
        return super().__getitem__(var)

    @define(Operation)
    def __setitem__(self, var: Variable[T], value: T | Variable[T] | Environment[T]) -> None:
        if var in self:
            raise ValueError(f"variable {var} already defined")
        super().__setitem__(var, value)


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

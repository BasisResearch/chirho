from typing import Generic, Iterable, Optional, Protocol, Type, TypeVar

import typing

from causal_pyro.effectful.ops.operations import StatefulInterpretation, Operation, define, register


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
    def keys(self) -> Iterable[Variable[T]]: ...


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


class EnvInterpretation(Generic[T], StatefulInterpretation[Environment[T], T]):
    state: Environment[T]

from typing import Generic, Optional, TypeVar

from ..ops.terms import Environment, Meta, Operation, Term, define


S, T = TypeVar("S"), TypeVar("T")


@define(Meta)
class Semiring(Generic[S]):
    pass


@define(Operation)
def zero(semiring: Semiring[S]) -> S:
    ...


@define(Operation)
def plus(semiring: Semiring[S], lhs: S, rhs: S) -> S:
    ...


@define(Operation)
def one(semiring: Semiring[S]) -> S:
    ...


@define(Operation)
def times(semiring: Semiring[S], lhs: S, rhs: S) -> S:
    ...


@define(Meta)
class Graded(Generic[S, T]):
    pass


@define(Operation)
def get_weight(value: Graded[S, T]) -> S:
    ...


@define(Operation)
def get_value(value: Graded[S, T]) -> T:
    ...


@define(Operation)
def contract(semiring: Semiring[S], value: Graded[S, T], ctx: Environment[Graded[S, T]]) -> Graded[S, T]:
    ...

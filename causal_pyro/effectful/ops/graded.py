from typing import Generic, Optional, TypeVar

from ..ops.terms import Context, Kind, Operation, Term, define


S, T = TypeVar("S"), TypeVar("T")


@define(Kind)
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


@define(Kind)
class Graded(Generic[S, T]):
    pass


GradedContext = Context[Graded[S, T]]


@define(Operation)
def grade(value: T, weight: Optional[S] = None) -> Graded[S, T]:
    ...


@define(Operation)
def get_weight(value: Graded[S, T]) -> S:
    ...


@define(Operation)
def get_value(value: Graded[S, T]) -> T:
    ...


@define(Operation)
def count(value: Graded[S, T], ctx: GradedContext[S, T]) -> Graded[S, T]:
    ...  # graded substitution

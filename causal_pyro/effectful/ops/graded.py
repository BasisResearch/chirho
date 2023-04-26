from typing import Generic, Optional, TypeVar

from ..ops.terms import Context, Kind, Operation, Term, define


S, T = TypeVar("S"), TypeVar("T")


@define(Kind)
class Graded(Generic[S, T]):
    pass


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
def get_base(value: Graded[S, T]) -> Graded[S, T]:
    ...


GradedContext = Context[Graded[S, T]]


@define(Operation)
def count(value: Graded[S, T], ctx: GradedContext[S, T]) -> Graded[S, T]:
    ...  # graded substitution

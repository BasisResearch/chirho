from typing import Generic, List, Optional, TypeVar

from .terms import Operation, Term, Interpretation, Environment, define, get_name, read


S, T = TypeVar("S"), TypeVar("T")


@define(Interpretation)
class FreeInterpretation(Generic[T]):
    pass


@define(Interpretation)
class HostInterpretation(Generic[T]):
    pass


@define(Operation)
def compose(interpretation: Interpretation[T], other: Interpretation[T]) -> Interpretation[T]:
    ...


@define(Operation)
def cont(ctx: Environment[T], result: Optional[T]) -> T:
    ...


@define(Operation)
def product(interpretation: Interpretation[T], cointerpretation: Interpretation[T]) -> Interpretation[T]:
    ...


@define(Operation)
def reflect(ctx: Environment[T], result: Optional[T]) -> T:
    ...

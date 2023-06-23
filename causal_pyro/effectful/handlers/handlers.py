from typing import Generic, Hashable, Optional, TypeVar, List

from ..ops.terms import Environment, Operation, Term, define, get_args, get_head
from ..ops.interpretations import Interpretation, union, product, compose, quotient
from ..ops.metacircular import evaluate, apply


S, T = TypeVar("S"), TypeVar("T")


@define(Environment)
class BaseEnvironment(Generic[T], dict[Hashable, T]):
    pass


@define(Interpretation)
class BaseInterpretation(Generic[T], BaseEnvironment[Operation[T]]):
    pass


@define(evaluate)
def evaluate_union_interpretation(ctx: Environment[T], interpretation: BaseInterpretation, term: Term[T]) -> T:
    op = get_head(term)
    for subinterpretation in interpretation.interpretations:
        if op in subinterpretation:
            return evaluate(ctx, subinterpretation, term)
    return term


class UnionInterpretation(BaseInterpretation):
    interpretations: List[BaseInterpretation]  


@define(union)
def union_interpretation(interpretation: BaseInterpretation, other: BaseInterpretation) -> UnionInterpretation:
    return UnionInterpretation(interpretations=[interpretation, other])


class ComposeInterpretation(BaseInterpretation):
    interpretations: List[BaseInterpretation]


@define(compose)
def compose_interpretation(interpretation: BaseInterpretation, other: BaseInterpretation) -> ComposeInterpretation:
    return ComposeInterpretation(interpretations=[interpretation, other])


class ProductInterpretation(BaseInterpretation):
    interpretations: List[BaseInterpretation]


@define(product)
def product_interpretation(interpretation: BaseInterpretation, other: BaseInterpretation) -> ProductInterpretation:
    return ProductInterpretation(interpretations=[interpretation, other])

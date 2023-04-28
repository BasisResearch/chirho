from typing import Generic, Optional, TypeVar, List

from ..ops.terms import Environment, Operation, Term, define, get_args, get_head
from ..ops.interpretations import Interpretation, union, product, compose, quotient
from ..ops.interpreter import evaluate, apply


S, T = TypeVar("S"), TypeVar("T")


@define(Interpretation)
class BaseModel:
    pass


class UnionModel(BaseModel):
    models: List[BaseModel]


@define(union)
def union_model(model: BaseModel, other: BaseModel) -> UnionModel:
    return UnionModel(models=[model, other])


@define(evaluate)
def evaluate_union_model(ctx: Environment[T], model: UnionModel, term: Term[T]) -> T:
    op = get_head(term)
    for submodel in model.models:
        if op in submodel:
            return evaluate(ctx, submodel, term)
    return term


class ComposeModel(BaseModel):
    models: List[BaseModel]


@define(compose)
def compose_model(model: BaseModel, other: BaseModel) -> ComposeModel:
    return ComposeModel(models=[model, other])


@define(evaluate)
def evaluate_compose_model(ctx: Environment[T], model: ComposeModel, term: Term[T]) -> T:
    for model in reversed(model.models):
        term = evaluate(ctx, model, term)
    return term


class ProductModel(BaseModel):
    models: List[BaseModel]


@define(product)
def product_model(model: BaseModel, other: BaseModel) -> ProductModel:
    return ProductModel(models=[model, other])


@define(evaluate)
def evaluate_product_model(ctx: Environment[T], model: ProductModel, term: Term[T]) -> T:
    return evaluate(ctx, inl(model), term), evaluate(ctx, inr(model), term)


class QuotientModel(BaseModel):
    models: List[BaseModel]


@define(quotient)
def quotient_model(model: BaseModel, other: BaseModel) -> QuotientModel:
    return QuotientModel(models=[model, other])


@define(evaluate)
def evaluate_quotient_model(ctx: Environment[T], model: QuotientModel, term: Term[T]) -> T:
    for model in model.models:
        term = evaluate(ctx, model, term)
    return term

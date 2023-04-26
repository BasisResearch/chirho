from typing import Generic, Optional, TypeVar, List

from ..ops.terms import Context, Operation, Term, define
from ..ops.models import Model, evaluate, apply, union, product, compose, quotient


S, T = TypeVar("S"), TypeVar("T")


@define(Model)
class BaseModel:
    pass


class UnionModel(BaseModel):
    models: List[BaseModel]


@default_register(union)
def union_model(model: BaseModel, other: BaseModel) -> UnionModel:
    return UnionModel(models=[model, other])


@default_register(evaluate)
def evaluate_union_model(ctx: Context[T], model: UnionModel, term: Term[T]) -> T:
    for model in model.models:
        term = evaluate(ctx, model, term)
    return term


class ComposeModel(BaseModel):
    models: List[BaseModel]


@default_register(compose)
def compose_model(model: BaseModel, other: BaseModel) -> ComposeModel:
    return ComposeModel(models=[model, other])


@default_register(evaluate)
def evaluate_compose_model(ctx: Context[T], model: ComposeModel, term: Term[T]) -> T:
    for model in reversed(model.models):
        term = evaluate(ctx, model, term)
    return term


class ProductModel(BaseModel):
    models: List[BaseModel]


@default_register(product)
def product_model(model: BaseModel, other: BaseModel) -> ProductModel:
    return ProductModel(models=[model, other])


@default_register(evaluate)
def evaluate_product_model(ctx: Context[T], model: ProductModel, term: Term[T]) -> T:
    return evaluate(ctx, inl(model), term), evaluate(ctx, inr(model), term)


class QuotientModel(BaseModel):
    models: List[BaseModel]


@default_register(quotient)
def quotient_model(model: BaseModel, other: BaseModel) -> QuotientModel:
    return QuotientModel(models=[model, other])


@default_register(evaluate)
def evaluate_quotient_model(ctx: Context[T], model: QuotientModel, term: Term[T]) -> T:
    for model in model.models:
        term = evaluate(ctx, model, term)
    return term

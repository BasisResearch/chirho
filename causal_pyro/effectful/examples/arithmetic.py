from typing import Any, Callable, Container, ContextManager, Generic, List, NamedTuple, Optional, Type, TypeVar, Union

from ..ops.terms import Atom, Environment, Operation, Term, define
from ..ops.interpretations import Interpretation, evaluate, cont, reflect


S, T = TypeVar("S"), TypeVar("T")


@define(Operation)
def add(x: int, y: int) -> int:
    ...


@define(Operation)
def lit(x: int) -> int:
    ...


if __name__ == "__main__":
    print(evaluate(add(lit(1), lit(2))))

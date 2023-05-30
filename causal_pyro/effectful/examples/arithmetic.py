from typing import Any, Callable, Container, ContextManager, Generic, List, NamedTuple, Optional, Type, TypeVar, Union

from ..ops.terms import Atom, Environment, Operation, Term, define
from ..ops.interpretations import Interpretation, evaluate, cont, reflect


S, T = TypeVar("S"), TypeVar("T")


@define(Operation)
def lit(x: int) -> int:
    ...


@define(Operation)
def add(x: int, y: int) -> int:
    return x + y


@define(Operation)
def mul(x: int, y: int) -> int:
    return x * y


@define(Operation)
def addmul(x: int, y: int, z: int) -> int:
    return mul(add(x, y), z)


@define(Interpretation)(add)
def add_print(x: int, y: int) -> int:
    print("overload add")
    return x + y


if __name__ == "__main__":
    print(add(1, 2) * mul(3, 4))
    print(addmul(1, 2, 3))
    print(evaluate(add(lit(1), lit(2))))

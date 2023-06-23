from typing import Callable

import functools

from causal_pyro.effectful.ops.operations import Interpretation, Operation, define
from causal_pyro.effectful.ops.interpretations import \
    compose, fwd, handler, product, reflect
from causal_pyro.effectful.ops.terms import Term, Variable, LazyInterpretation, head_of, args_of


@define(Operation)
def add(x: int, y: int) -> int:
    print("C")
    return x + y


@define(Operation)
def add3(x: int, y: int, z: int) -> int:
    return add(x, add(y, z))


@define(Operation)
def mul(x: int, y: int) -> int:
    return x * y


def print_wrap(fn: Callable, name="A"):
    def wrapped(result, *args, **kwargs):
        print(f"{name} calling {fn} with {args} {kwargs}")
        result = fwd(result)
        assert result is not None, "result is None" 
        print(f"result: {result}")
        return result
    return wrapped


if __name__ == "__main__":

    printme1 = define(Interpretation)({
        add: print_wrap(add.body, name="A"),
    })

    printme2 = define(Interpretation)({
        add: print_wrap(add.body, name="B"),
    })

    default = define(Interpretation)({add: add.default})

    print(add(3, 4))
    print(add3(3, 4, 5))

    with handler(default) as h:
        print(add(3, 4))
        print(add3(3, 4, 5))

    with handler(compose(default, printme1)):
        print(add(3, 4))
        print(add3(3, 4, 5))

    with handler(default), handler(printme1):
        print(add(3, 4))
        print(add3(3, 4, 5))

    with handler(default), handler(compose(printme1, printme2)):
        print(add(3, 4))
        print(add3(3, 4, 5))

    with handler(compose(printme1, printme2)):
        print(add(3, 4))
        print(add3(3, 4, 5))

    with handler(compose(default, printme1, printme2)) as h:
        print(add(3, 4))
        print(add3(3, 4, 5))

    with handler(product(default, printme2)) as h:
        print(add(3, 4))

    with handler(LazyInterpretation(add)) as h:
        print(add(3, 4))
        print(add3(3, 4, 5))

    with handler(compose(LazyInterpretation(add), printme2)) as h:
        print(add(3, 4))
        print(add3(3, 4, 5))

    with handler(compose(LazyInterpretation(add, add3), printme2)) as h:
        print(add(3, 4))
        print(add3(3, 4, 5))

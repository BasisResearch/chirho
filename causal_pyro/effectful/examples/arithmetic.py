from typing import Callable

import functools

from causal_pyro.effectful.ops.bootstrap import Interpretation, Operation, define
from causal_pyro.effectful.ops.interpretations import \
    compose, fwd, handler, \
    product, reflect, runner


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
        print(f"result: {result}")
        return result
    return functools.wraps(fn)(wrapped)


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

    # with handler(printme2) as h:
    #     print(add(3, 4))
    #     print(add3(3, 4, 5))

    # with handler(compose(printme1, printme2)) as h:
    #     print(add(3, 4))
    #     print(add3(3, 4, 5))

    # with handler(compose(default, printme1, printme2)) as h:
    #     print(add(3, 4))
    #     print(add3(3, 4, 5))

    with runner(compose(default, printme1)) as r:
        with handler(printme2) as h:
            print(add(3, 4))

from typing import Any, Callable, Container, ContextManager, Generic, List, NamedTuple, Optional, Type, TypeVar, Union

from ..ops.bootstrap import Interpretation, Operation, define
from ..ops.interpretations import \
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

def print_wrap(fn, name="A"):
    def wrapped(result, *args, **kwargs):
        print(f"{name} calling {fn} with {args} {kwargs}")
        result = fwd(result)
        print(f"result: {result}")
        return result
    return wrapped


if __name__ == "__main__":

    printme1 = Interpretation({
        add: print_wrap(add.body, name="A"),
    })

    printme2 = Interpretation({
        add: print_wrap(add.body, name="B"),
    })

    default = Interpretation({add: lambda res, *args: res if res is not None else add.body(*args)})

    # printme3 = compose(compose(default, printme1), printme2)
    # printme3 = compose(default, compose(printme1, printme2))
    printme3 = compose(default, printme1, printme2)

    print(add(3, 4))
    print(add3(3, 4, 5))

    with handler(printme3) as h:
        print(add(3, 4))
        print(add3(3, 4, 5))

    # what should happen with runner?
    # when reflect is called, it should jump to the next runner
    # i.e. re-invoke the operation under the next product interpretation

    with runner(printme1) as r:
        with handler(printme2) as h:
            print(add(3, 4))


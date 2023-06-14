from typing import Callable, Generic, Hashable, List, Mapping, Optional, Protocol, Type, TypeVar

import functools


T = TypeVar("T")

Interpretation = Mapping[Hashable, Callable[..., T]]


class Operation(Generic[T]):

    def __init__(self, body: Callable[..., T]):
        self.body = body

    def __call__(self, *args: T, **kwargs: T) -> T:
        try:
            interpret = get_interpretation()[self]
            args = (None,) + args
        except KeyError:
            interpret = self.body
        except NameError as e:  # bootstrapping case
            if e.args[0] == "name 'get_interpretation' is not defined":
                interpret = self.body if self.body is not Operation else lambda x: x
            else:
                raise
        return interpret(*args, **kwargs)


class Runtime(Generic[T]):
    interpretation: Interpretation[T]

    def __init__(self, interpretation: Interpretation[T]):
        self.interpretation = interpretation


RUNTIME = Runtime(dict())


@Operation
@functools.cache
def define(m: Type[T]) -> T | Type[T] | Callable[..., T] | Callable[..., Callable[..., T]]:
    # define is the embedding function from host syntax to embedded syntax
    return Operation(m) if m is Operation else define(Operation)(m)


@Operation
def get_interpretation() -> Interpretation[T]:
    return RUNTIME.interpretation


@Operation
def swap_interpretation(intp: Interpretation[T]) -> Interpretation[T]:
    old_intp = RUNTIME.interpretation
    RUNTIME.interpretation = intp
    return old_intp

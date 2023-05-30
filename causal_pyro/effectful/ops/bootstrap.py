from typing import Any, Callable, Container, ContextManager, Generic, Hashable, Mapping, NamedTuple, Protocol, Optional, Set, Type, TypeVar, Union, runtime_checkable

S, T = TypeVar("S"), TypeVar("T")


class Operation(Generic[T]):

    def __init__(self, body: Callable[..., T]):
        self.body = body

    def __call__(self, *args: T, **kwargs: T) -> T:
        return apply(self, *args, **kwargs)


def get_op_body(op: Operation[T]) -> Callable[..., T]:
    return op.body


def apply(op: Operation[T], *args: T, **kwargs: T) -> T:
    return get_op_body(op)(*args, **kwargs)


def define_define(constructors: Mapping[Hashable, Callable]):

    # define is the embedding function from host syntax to embedded syntax
    def define(m: Type[T] | Hashable) -> Callable[..., Type[T]]:
        return constructors.get(m, m)

    def metadef(m: Type[T] | Hashable, fn: Optional[Callable[..., Type[T]]] = None):
        if fn is None:  # curry
            return lambda fn: metadef(m, fn)

        constructors[m] = define(Operation)(fn)
        return define(m)

    metadef(define, metadef)
    return define


define = define_define({})


@define(define)(Operation)
def defop(fn: Callable[..., T]) -> Operation[T]:
    return Operation(fn)


# TODO self-hosting
# apply = define(Operation)(apply)
# get_op_body = define(Operation)(get_op_body)

from typing import Callable, ClassVar, Generic, Hashable, Iterable, List, Mapping, Optional, Protocol, Type, TypeVar, runtime_checkable

import functools


S = TypeVar("S")
T = TypeVar("T")


@runtime_checkable
class Operation(Protocol[T]):
    def __call__(self, *args: T, **kwargs) -> T: ...


class BaseOperation(Generic[T]):

    def __init__(self, body: Callable[..., T]):
        self.body = body

    @property
    def default(self) -> Callable[..., T]:
        return functools.wraps(self.body)(
            lambda res, *args, **kwargs: res if res is not None else self.body(*args, **kwargs)
        )

    def __call__(self, *args: T, **kwargs: T) -> T:
        args = (None,) + args
        try:
            interpret = get_interpretation()[self]
        except KeyError:
            interpret = self.default
        except NameError as e:
            if e.args[0] == "name 'get_interpretation' is not defined":
                interpret = self.default if self.body is not BaseOperation else lambda _, x: x
            else:
                raise
        return interpret(*args, **kwargs)


@runtime_checkable
class Interpretation(Protocol[T]):
    def __setitem__(self, op: Operation[T], interpret: Callable[..., T]) -> None: ...
    def __getitem__(self, op: Operation[T]) -> Callable[..., T]: ...
    def __contains__(self, op: Operation[T]) -> bool: ...
    def keys(self) -> Iterable[Operation[T]]: ...


BaseInterpretation = dict[Operation[T], Callable[..., T]]


class Runtime(Generic[T]):
    interpretation: Interpretation[T]

    def __init__(self, interpretation: Interpretation[T]):
        self.interpretation = interpretation


RUNTIME = Runtime(BaseInterpretation())


@BaseOperation
@functools.cache
def define(m: Type[T]) -> Operation[T]:
    # define is the embedding function from host syntax to embedded syntax
    return BaseOperation(BaseOperation) if m is Operation else define(Operation)(m)


@define(Operation)
def get_interpretation() -> Interpretation[T]:
    return RUNTIME.interpretation


@define(Operation)
def swap_interpretation(intp: Interpretation[T]) -> Interpretation[T]:
    old_intp = RUNTIME.interpretation
    RUNTIME.interpretation = intp
    return old_intp


@define(Operation)
def register(
    op: Operation[T],
    intp: Optional[Interpretation[T]] = None,
    interpret_op: Optional[Callable[..., T]] = None
):
    if interpret_op is None:
        return functools.partial(register, op, intp=intp)

    if intp is None:
        if isinstance(op, BaseOperation):
            setattr(op, "body", interpret_op)
            return interpret_op
    elif isinstance(intp, Interpretation):
        intp.__setitem__(op, interpret_op)
        return interpret_op
    raise NotImplementedError(f"Cannot register {op} in {intp}")


register(define(Interpretation), None, BaseInterpretation)

from typing import Callable, ClassVar, Generic, Hashable, Iterable, List, Mapping, Optional, Protocol, Type, TypeVar, runtime_checkable

import functools
import weakref


S = TypeVar("S")
T = TypeVar("T")


@runtime_checkable
class Operation(Protocol[T]):
    def __call__(self, *args: T, **kwargs) -> T: ...


@runtime_checkable
class Interpretation(Protocol[T]):
    def __setitem__(self, op: Operation[T], interpret: Callable[..., T]) -> None: ...
    def __getitem__(self, op: Operation[T]) -> Callable[..., T]: ...
    # def __contains__(self, op: Operation[T]) -> bool: ...
    def keys(self) -> Iterable[Operation[T]]: ...


class BaseOperation(Generic[T]):

    def __init__(self, body: Callable[..., T]):
        self.body = body

    def __call__(self, *args: T, **kwargs: T) -> T:
        try:
            interpret = get_interpretation()[self]
            args = (None,) + args
        except KeyError:
            interpret = self.body
        except NameError as e:
            if e.args[0] == "name 'get_interpretation' is not defined":
                interpret = self.body if self.body is not BaseOperation else lambda x: x
            else:
                raise
        return interpret(*args, **kwargs)


class StatefulInterpretation(Generic[S, T]):
    state: S

    _op_interpretations: ClassVar[dict[Operation, Callable]] = weakref.WeakKeyDictionary()

    def __init_subclass__(cls) -> None:
        cls._op_interpretations: weakref.WeakKeyDictionary[Operation[T], Callable[..., T]] = \
            weakref.WeakKeyDictionary()
        return super().__init_subclass__()

    def __init__(self, state: S):
        self.state = state

    @classmethod
    def __setitem__(cls, op: Operation[T], interpret_op: Callable[..., T]) -> None:
        cls._op_interpretations[op] = interpret_op

    def __getitem__(self, op: Operation[T]) -> Callable[..., T]:
        return functools.partial(self._op_interpretations[op], self.state)

    @classmethod
    def keys(cls) -> Iterable[Operation[T]]:
        return cls._op_interpretations.keys()


class Runtime(Generic[T]):
    interpretation: Interpretation[T]

    def __init__(self, interpretation: Interpretation[T]):
        self.interpretation = interpretation


RUNTIME = Runtime(dict[Operation[T], Callable[..., T]]())


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
    intp: Optional[Interpretation[T] | Type[StatefulInterpretation[S, T]]],
    op: Operation[T],
    interpret_op: Optional[Callable[..., T]] = None
):
    if interpret_op is None:
        return functools.partial(register, intp, op)

    if intp is None:
        if isinstance(op, BaseOperation):
            setattr(op, "body", interpret_op)
            return interpret_op
    elif isinstance(intp, Interpretation):
        intp[op] = interpret_op
        return interpret_op
    elif issubclass(intp, StatefulInterpretation):
        intp.__setitem__(op, interpret_op)
        return interpret_op
    raise NotImplementedError(f"Cannot register {op} in {intp}")


register(None, define(Interpretation))(dict[Operation[T], Callable[..., T]])

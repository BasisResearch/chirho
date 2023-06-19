from typing import Callable, ClassVar, Generic, Hashable, Iterable, List, Mapping, Optional, Protocol, Type, TypedDict, TypeVar, runtime_checkable

import functools


S = TypeVar("S")
T = TypeVar("T")


@runtime_checkable
class Operation(Protocol[T]):
    def __call__(self, *args: T, **kwargs) -> T: ...


class _BaseOperation(Generic[T]):

    def __init__(self, body: Callable[..., T]):
        self.body = body

    @property
    def default(self) -> Callable[..., T]:
        return functools.wraps(self.body)(
            lambda res, *args, **kwargs: res if res is not None else self.body(*args, **kwargs)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({getattr(self.body, '__name__', self.body)}"

    def __call__(self, *args: T, **kwargs: T) -> T:
        return self.default(None, *args, **kwargs)


@_BaseOperation
@functools.cache
def define(m: Type[T]) -> Operation[T]:
    # define is the embedding function from host syntax to embedded syntax
    return _BaseOperation if m is Operation else define(Operation)(m)


@runtime_checkable
class Interpretation(Protocol[T]):
    def __setitem__(self, op: Operation[T], interpret: Callable[..., T]) -> None: ...
    def __getitem__(self, op: Operation[T]) -> Callable[..., T]: ...
    def __contains__(self, op: Operation[T]) -> bool: ...
    def keys(self) -> Iterable[Operation[T]]: ...


@define(Operation)
def register(
    op: Operation[T],
    intp: Optional[Interpretation[T]] = None,
    interpret_op: Optional[Callable[..., T]] = None
):
    if interpret_op is None:
        return lambda interpret_op: register(op, intp, interpret_op)

    if intp is None:
        if isinstance(op, _BaseOperation):
            setattr(op, "body", interpret_op)
            return interpret_op
    elif isinstance(intp, Interpretation):
        intp.__setitem__(op, interpret_op)
        return interpret_op
    raise NotImplementedError(f"Cannot register {op} in {intp}")


register(define(Interpretation), None, dict[Operation[T], Callable[..., T]])

#####################################################################


class Runtime(TypedDict):
    interpretation: Interpretation


@functools.cache
def get_runtime() -> Runtime:
    return Runtime(interpretation=define(Interpretation)())


@define(Operation)
def get_interpretation() -> Interpretation[T]:
    return get_runtime()["interpretation"]


@define(Operation)
def swap_interpretation(intp: Interpretation[T]) -> Interpretation[T]:
    old_intp = get_runtime()["interpretation"]
    get_runtime()["interpretation"] = intp
    return old_intp


# set cached initial runtime state
get_runtime()


# bootstrap - operations now use get_interpretation
@functools.partial(setattr, _BaseOperation, "__call__")
@functools.wraps(_BaseOperation.__call__)
def _op_call(op: Operation[T], *args: T, **kwargs) -> S:
    intp = op.default(None) if op is get_interpretation else get_interpretation()
    interpret = intp[op] if op in intp else getattr(op, "default")
    return interpret(None, *args, **kwargs)

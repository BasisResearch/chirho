from typing import Callable, ClassVar, Generic, Hashable, Iterable, List, Mapping, Optional, Protocol, Type, TypedDict, TypeVar, runtime_checkable

import functools


S = TypeVar("S")
T = TypeVar("T")


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


_BaseInterpretation = dict[Callable, Callable[..., T]]


class Runtime(TypedDict):
    interpretation: Mapping[Callable, Callable]


@functools.cache
def get_runtime() -> Runtime:
    return Runtime(interpretation=_BaseInterpretation())


@_BaseOperation
def get_interpretation() -> Mapping[Callable, Callable]:
    return get_runtime()["interpretation"]


@_BaseOperation
def swap_interpretation(intp: Mapping[Callable, Callable]) -> Mapping[Callable, Callable]:
    old_intp = get_runtime()["interpretation"]
    get_runtime()["interpretation"] = intp
    return old_intp


@functools.wraps(_BaseOperation.__call__)
def _op_call(op: Callable[..., T], *args: T, **kwargs) -> S:
    intp = op.default(None) if op is get_interpretation else get_interpretation()
    try:
        interpret = intp[op]
    except KeyError:
        interpret = getattr(op, "default")  # TODO abstract or codify default?
    return interpret(None, *args, **kwargs)

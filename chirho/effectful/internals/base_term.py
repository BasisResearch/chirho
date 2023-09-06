from typing import Generic, Iterable, Mapping, TypeVar

from chirho.effectful.ops.operation import Operation

S = TypeVar("S")
T = TypeVar("T")


class _BaseTerm(Generic[T]):
    __op__: Operation[..., T]
    __args__: tuple["_BaseTerm[T]" | T, ...]
    __kwargs__: dict[str, "_BaseTerm[T]" | T]

    def __init__(
        self,
        op: Operation[..., T],
        args: Iterable["_BaseTerm[T]" | T],
        kwargs: Mapping[str, "_BaseTerm[T]" | T]
    ):
        self.__op__ = op
        self.__args__ = tuple(args)
        self.__kwargs__ = dict(kwargs)

    def __repr__(self) -> str:
        return f"{self.__op__}(" + \
            f"{', '.join(map(repr, self.__args__))}," + \
            f"{', '.join(f'{k}={v}' for k, v in self.__kwargs__.items())})"

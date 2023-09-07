from typing import Generic, Iterable, Mapping, TypeVar

from chirho.effectful.ops.operation import Operation

S = TypeVar("S")
T = TypeVar("T")


class _BaseTerm(Generic[T]):
    op: Operation[..., T]
    args: tuple["_BaseTerm[T]" | T, ...]
    kwargs: dict[str, "_BaseTerm[T]" | T]

    def __init__(
        self,
        __op: Operation[..., T],
        __args: Iterable["_BaseTerm[T]" | T],
        __kwargs: Mapping[str, "_BaseTerm[T]" | T]
    ):
        self.op = __op
        self.args = tuple(__args)
        self.kwargs = dict(__kwargs)

    def __repr__(self) -> str:
        return f"{self.op}(" + \
            f"{', '.join(map(repr, self.args))}," + \
            f"{', '.join(f'{k}={v}' for k, v in self.kwargs.items())})"

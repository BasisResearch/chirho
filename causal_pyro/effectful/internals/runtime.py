from typing import Callable, Mapping, TypedDict, TypeVar

import functools

S = TypeVar("S")
T = TypeVar("T")


class Runtime(TypedDict):
    interpretation: Mapping[Callable, Callable]


@functools.lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return Runtime(interpretation=dict[Callable, Callable]())


def get_interpretation() -> Mapping[Callable, Callable]:
    return get_runtime()["interpretation"]


def swap_interpretation(
    intp: Mapping[Callable[..., S], Callable[..., T]]
) -> Mapping[Callable[..., S], Callable[..., T]]:
    old_intp = get_runtime()["interpretation"]
    get_runtime()["interpretation"] = intp
    return old_intp

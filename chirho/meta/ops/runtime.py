import dataclasses
import functools
from typing import Callable, Generic, Mapping, TypeVar

S = TypeVar("S")
T = TypeVar("T")
_Intp = TypeVar("_Intp", bound=Mapping[Callable, Callable])


@dataclasses.dataclass
class Runtime(Generic[_Intp]):
    interpretation: _Intp


@functools.lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return Runtime(interpretation={})


def get_interpretation():
    return get_runtime().interpretation


def swap_interpretation(intp: _Intp) -> _Intp:
    old_intp = get_runtime().interpretation
    get_runtime().interpretation = intp
    return old_intp

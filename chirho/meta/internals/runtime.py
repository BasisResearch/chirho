import dataclasses
import functools
import weakref
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


def weak_memoize(f: Callable[[S], T]) -> Callable[[S], T]:
    """
    Memoize a one-argument function using a dictionary
    whose keys are weak references to the arguments.
    """

    cache: weakref.WeakKeyDictionary[S, T] = weakref.WeakKeyDictionary()

    @functools.wraps(f)
    def wrapper(x: S) -> T:
        try:
            return cache[x]
        except KeyError:
            result = f(x)
            cache[x] = result
            return result

    return wrapper

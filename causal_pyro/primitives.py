from lib2to3.pgen2.token import OP
import pyro

from typing import Callable, Optional, Union, TypeVar


T = TypeVar("T")

Intervention = Union[
    Optional[T],
    Callable[[T], T],
]


@pyro.poutine.runtime.effectful
def intervene(
    obs: T,
    act: Intervention[T] = None,
    *,
    event_dim: Optional[int] = None
) -> T:
    """
    Intervene on a value in a probabilistic program.
    """
    if callable(act):
        return act(obs)
    elif act is None:
        return obs
    return act
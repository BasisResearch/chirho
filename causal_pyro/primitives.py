from typing import Callable, Optional, TypeVar, Union

import pyro

T = TypeVar("T")

Intervention = Union[
    Optional[T],
    Callable[[T], T],
]


@pyro.poutine.runtime.effectful
def intervene(
    obs: T, act: Intervention[T] = None, *, event_dim: Optional[int] = None
) -> T:
    """
    Intervene on a value in a probabilistic program.
    """
    if callable(act):
        return act(obs)
    elif act is None:
        return obs
    return act

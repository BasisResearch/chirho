from __future__ import annotations

from typing import TypeVar

from chirho.dynamical.ops import State
from chirho.interventional.handlers import intervene

T = TypeVar("T")


@intervene.register(State)
def state_intervene(obs: State[T], act: State[T], **kwargs) -> State[T]:
    new_state: State[T] = State()
    for k in obs.keys:
        setattr(
            new_state, k, intervene(getattr(obs, k), getattr(act, k, None), **kwargs)
        )
    return new_state

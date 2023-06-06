from typing import Optional, Tuple, TypeVar

import pyro

from causal_pyro.indexed.ops import IndexSet, cond, scatter
from causal_pyro.interventional.ops import Intervention, intervene

S, T = TypeVar("S"), TypeVar("T")


@pyro.poutine.runtime.effectful(type="split")
@pyro.poutine.block(hide_types=["intervene"])
def split(obs: T, acts: Tuple[Intervention[T], ...], **kwargs) -> T:
    """
    Split the state of the world at an intervention.
    """
    name = kwargs.get("name", None)
    act_values = {IndexSet(**{name: {0}}): obs}
    for i, act in enumerate(acts):
        act_values[IndexSet(**{name: {i + 1}})] = intervene(obs, act, **kwargs)

    return scatter(act_values, event_dim=kwargs.get("event_dim", 0))


@pyro.poutine.runtime.effectful(type="preempt")
@pyro.poutine.block(hide_types=["intervene"])
def preempt(obs: T, acts: Tuple[Intervention[T], ...], case: Optional[S] = None, **kwargs) -> T:
    """
    Effectful primitive operation for preempting values in a probabilistic program.
    """
    if case is None:
        return obs

    name = kwargs.get("name", None)
    act_values = {IndexSet(**{name: {0}}): obs}
    for i, act in enumerate(acts):
        act_values[IndexSet(**{name: {i + 1}})] = intervene(obs, act, **kwargs)

    return cond(act_values, case, event_dim=kwargs.get("event_dim", 0))

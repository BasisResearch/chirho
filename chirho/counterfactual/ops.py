from typing import Optional, Tuple, TypeVar

import pyro

from chirho.indexed.ops import IndexSet, cond, scatter
from chirho.interventional.ops import Intervention, intervene

S = TypeVar("S")
T = TypeVar("T")


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
def preempt(
    obs: T, acts: Tuple[Intervention[T], ...], case: Optional[S] = None, **kwargs
) -> T:
    """
    Effectful primitive operation for "preempting" values in a probabilistic program.

    Unlike the counterfactual operation :func:`~chirho.counterfactual.ops.split`,
    which returns multiple values concatenated along a new axis
    via the operation :func:`~chirho.indexed.ops.scatter`,
    :func:`preempt` returns a single value determined by the argument ``case``
    via :func:`~chirho.indexed.ops.cond` .

    In a probabilistic program, a :func:`preempt` call induces a mixture distribution
    over downstream values, whereas :func:`split` would induce a joint distribution.

    :param obs: The observed value.
    :param acts: The interventions to apply.
    :param case: The case to select.
    """
    if case is None:
        return obs

    name = kwargs.get("name", None)
    act_values = {IndexSet(**{name: {0}}): obs}
    for i, act in enumerate(acts):
        act_values[IndexSet(**{name: {i + 1}})] = intervene(obs, act, **kwargs)

    return cond(act_values, case, event_dim=kwargs.get("event_dim", 0))

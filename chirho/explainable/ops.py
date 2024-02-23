import functools
from typing import Optional, Tuple, TypeVar

import pyro

from chirho.indexed.ops import IndexSet, cond_n, gather
from chirho.interventional.ops import Intervention, intervene

S = TypeVar("S")
T = TypeVar("T")


@pyro.poutine.runtime.effectful(type="preempt")
@functools.partial(pyro.poutine.block, hide_types=["intervene"])
def preempt(
    obs: T, acts: Tuple[Intervention[T], ...], case: Optional[S] = None, **kwargs
) -> T:
    """
    Effectful primitive operation for "preempting" values in a probabilistic program.

    Unlike the counterfactual operation :func:`~chirho.counterfactual.ops.split`,
    which returns multiple values concatenated along a new axis
    via the operation :func:`~chirho.indexed.ops.scatter`,
    :func:`preempt` returns a single value determined by the argument ``case``
    via :func:`~chirho.indexed.ops.cond`.

    In a probabilistic program, a :func:`preempt` call induces a mixture distribution
    over downstream values, whereas :func:`split` would induce a joint distribution.

    :param obs: The observed value.
    :param acts: The interventions to apply.
    :param case: The case to select.
    """
    if case is None:
        return obs

    name = kwargs.get("name", None)
    obs_idx = IndexSet(**{name: {0}})
    act_values = {obs_idx: gather(obs, obs_idx, **kwargs)}
    for i, act in enumerate(acts):
        act_idx = IndexSet(**{name: {i + 1}})
        act_values[act_idx] = gather(
            intervene(act_values[obs_idx], act, **kwargs), act_idx, **kwargs
        )

    return cond_n(act_values, case, event_dim=kwargs.get("event_dim", 0))
